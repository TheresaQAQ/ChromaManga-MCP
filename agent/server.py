"""FastAPI 应用：托管前端 + /upload 文件接收 + /image 回图 + /ws WebSocket 流式。

启动流程：
  1. 校验配置 (validate)
  2. 预热 MCP Server（拉起 subprocess → 加载 SDXL → list_tools）
  3. 构造 Agent
  4. 监听 HTTP
"""
import asyncio
import json
import logging
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel

from . import config
from .graph import build_agent

logger = logging.getLogger(__name__)

# 识别工具返回文本里的图片路径（Windows / Unix 双栈）
_IMAGE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/](?:[^\s<>\"'`|?*\n]+?)\.(?:png|jpg|jpeg|webp)"
    r"|/(?:[^\s<>\"'`|?*\n]+?)\.(?:png|jpg|jpeg|webp))",
    re.IGNORECASE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("配置错误: %s", e)
        raise RuntimeError("Agent 启动配置不完整，见上方日志")

    logger.info("预热 MCP 连接（首次 30~60 秒，需加载 SDXL+ControlNet+LoRA）...")
    await build_agent()
    logger.info("Agent 预热完成，HTTP 服务已就绪")
    yield
    logger.info("Agent shutdown")


app = FastAPI(title="ChromaManga Agent", lifespan=lifespan)


# ─── Static frontend ──────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))


@app.get("/styles.css")
async def styles():
    return FileResponse(str(config.FRONTEND_DIR / "styles.css"), media_type="text/css")


@app.get("/app.js")
async def app_js():
    return FileResponse(
        str(config.FRONTEND_DIR / "app.js"), media_type="application/javascript"
    )


@app.get("/config")
async def client_config():
    """前端启动时拉取，用于渲染 Header（mock 徽章、模型名、工具数）。"""
    return {
        "mode": config.AGENT_MCP_MODE,
        "model": config.LLM_MODEL,
        "tool_total": 7,  # STEPS_ORDER 中用于进度显示的核心步骤数
    }


# ── LLM 配置 ─────────────────────────────────────────────────────

def _mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


@app.get("/llm-config")
async def get_llm_config():
    """前端打开 LLM 配置抽屉时调，回显当前配置（API Key 脱敏）。"""
    return {
        "base_url": config.LLM_BASE_URL,
        "model": config.LLM_MODEL,
        "api_key_masked": _mask_api_key(config.LLM_API_KEY),
        "has_api_key": bool(config.LLM_API_KEY),
        "temperature": config.LLM_TEMPERATURE,
    }


class LLMConfigPayload(BaseModel):
    base_url: str
    model: str
    api_key: str = ""   # 留空表示沿用当前 key（用户只想换模型时方便）
    persist: bool = False


@app.post("/llm-config")
async def set_llm_config(payload: LLMConfigPayload):
    """热切换 LLM。空 api_key 表示沿用当前；persist=True 同步写入 .env.agent。"""
    from .graph import rebuild_agent_with_llm

    api_key = (payload.api_key or "").strip() or config.LLM_API_KEY
    base_url = payload.base_url.strip()
    model = payload.model.strip()

    if not api_key:
        raise HTTPException(400, "缺少 api_key（首次配置必填）")
    if not base_url or not model:
        raise HTTPException(400, "base_url 和 model 不能为空")

    try:
        await rebuild_agent_with_llm(api_key=api_key, base_url=base_url, model=model)
    except Exception as e:
        logger.exception("rebuild_agent_with_llm 失败")
        raise HTTPException(500, f"重建 Agent 失败: {e}")

    persisted = False
    if payload.persist:
        try:
            _persist_llm_to_env_agent(api_key, base_url, model)
            persisted = True
        except Exception as e:
            logger.warning(".env.agent 写入失败: %s", e)

    return {
        "ok": True,
        "model": model,
        "base_url": base_url,
        "api_key_masked": _mask_api_key(api_key),
        "persisted": persisted,
    }


def _persist_llm_to_env_agent(api_key: str, base_url: str, model: str) -> None:
    """更新 .env.agent 中**未注释的** LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 三行。

    保留原文件其他内容（注释、AGENT_MCP_MODE、AGENT_PORT 等）。
    若文件中没有该字段则追加到末尾。
    """
    env_path = config.PROJECT_ROOT / ".env.agent"
    if not env_path.exists():
        return

    updates = {"LLM_API_KEY": api_key, "LLM_BASE_URL": base_url, "LLM_MODEL": model}
    seen: set[str] = set()
    new_lines: list[str] = []

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            new_lines.append(line)
            continue
        m = re.match(r"^([A-Z_]+)\s*=", line)
        if m and m.group(1) in updates and m.group(1) not in seen:
            new_lines.append(f"{m.group(1)}={updates[m.group(1)]}")
            seen.add(m.group(1))
        else:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ─── Image upload / serve ─────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """接收前端上传的图片，保存到 data/inputs/，返回绝对路径供 Agent 使用。"""
    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise HTTPException(400, f"不支持的图片格式: {suffix}")

    save_path = config.UPLOAD_DIR / f"upload_{int(time.time() * 1000)}{suffix}"
    content = await file.read()
    save_path.write_bytes(content)
    abs_path = str(save_path.resolve())
    logger.info("图片已保存: %s (%d bytes)", abs_path, len(content))
    return {"path": abs_path, "url": f"/image?path={abs_path}"}


@app.post("/example-upload")
async def example_upload():
    """一键使用仓库内 `examples/c1a670e96.../00_input.png` 作为上传图。

    Mock 模式下，各阶段 mock 工具会从同目录拷贝真实效果图，整条流程就能看到
    原汁原味的"漫画 → 线稿 → 检测 → 上色 → 还原"视觉产物。
    """
    src = (
        config.PROJECT_ROOT
        / "examples"
        / "c1a670e964087446163d32eeab823613_regional"
        / "00_input.png"
    )
    if not src.exists():
        raise HTTPException(404, f"示例图不存在: {src}")
    dst = config.UPLOAD_DIR / f"example_{int(time.time() * 1000)}.png"
    shutil.copyfile(src, dst)
    abs_path = str(dst.resolve())
    logger.info("示例图已复制为输入: %s", abs_path)
    return {"path": abs_path, "url": f"/image?path={abs_path}"}


@app.get("/image")
async def get_image(path: str):
    """按绝对路径返回任意图片（含 MCP 产出的中间产物）。"""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"文件不存在: {path}")
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media)


# ─── WebSocket streaming ──────────────────────────────────────────────

def _extract_image_paths(text: str) -> list[str]:
    """从工具返回文本里抓取存在的图片路径，去重保序。"""
    seen: set[str] = set()
    result: list[str] = []
    for m in _IMAGE_PATH_RE.finditer(text):
        path = m.group(0)
        try:
            if Path(path).exists() and path not in seen:
                seen.add(path)
                result.append(path)
        except OSError:
            continue
    return result


async def _send_updates_message(ws: WebSocket, msg) -> None:
    """处理 updates 流的完整消息：仅推 tool_call / tool_result 给前端。

    AIMessage.content 改由 messages 流以 token 级增量推送（type='ai'），避免重复。
    """
    if isinstance(msg, AIMessage):
        for tc in msg.tool_calls or []:
            await ws.send_json(
                {
                    "type": "tool_call",
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                }
            )
    elif isinstance(msg, ToolMessage):
        content = str(msg.content) if msg.content is not None else ""
        await ws.send_json(
            {
                "type": "tool_result",
                "id": msg.tool_call_id,
                "content": content[:6000],
                "images": _extract_image_paths(content),
            }
        )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    agent = await build_agent()
    history: list = []

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            # 客户端重置会话：清空服务端 history，保持 ws 连接
            if data.get("type") == "reset":
                history.clear()
                await ws.send_json({"type": "reset_done"})
                continue

            user_text = (data.get("content") or "").strip()
            image_path = data.get("image_path")

            if not user_text and not image_path:
                continue

            injected = user_text
            if image_path:
                injected = (
                    f"{user_text}\n\n[已上传图片，绝对路径: {image_path}]"
                    if user_text
                    else f"用户上传了一张漫画图，请开始上色流程。\n\n[已上传图片，绝对路径: {image_path}]"
                )

            history.append(HumanMessage(content=injected))

            try:
                # 双流：messages 拿 LLM token 级 chunks，updates 拿节点级完整消息
                async for stream_event in agent.astream(
                    {"messages": history},
                    {"recursion_limit": config.AGENT_RECURSION_LIMIT},
                    stream_mode=["updates", "messages"],
                ):
                    if not isinstance(stream_event, tuple) or len(stream_event) != 2:
                        continue
                    mode, payload = stream_event

                    if mode == "messages":
                        # payload = (AIMessageChunk | ToolMessage, metadata)
                        if not isinstance(payload, tuple) or not payload:
                            continue
                        message_chunk = payload[0]
                        if isinstance(message_chunk, AIMessageChunk):
                            text = message_chunk.content
                            # content 可能是 str 或 list[dict]（多模态），这里只处理纯文本 token
                            if isinstance(text, str) and text:
                                await ws.send_json({"type": "ai", "content": text})

                    elif mode == "updates":
                        # payload = {node_name: {"messages": [...]}}
                        if not isinstance(payload, dict):
                            continue
                        for node_update in payload.values():
                            if not isinstance(node_update, dict):
                                continue
                            for m in node_update.get("messages", []) or []:
                                history.append(m)
                                await _send_updates_message(ws, m)
            except Exception as e:
                logger.exception("Agent 执行异常")
                await ws.send_json({"type": "error", "content": f"Agent 执行出错: {e}"})

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception as e:
        logger.exception("WebSocket 处理器异常")
        try:
            await ws.send_json({"type": "error", "content": str(e)})
            await ws.close()
        except Exception:
            pass
