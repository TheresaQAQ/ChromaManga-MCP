"""ChromaManga Agent 入口：启动 FastAPI + uvicorn，自动拉起 MCP Server。

用法（在已装好 ChromaManga + MCP + Agent 依赖的机器上）：

    # 1. 准备 LLM 凭据
    cp .env.agent.example .env.agent
    # 编辑 .env.agent 填入 LLM_API_KEY

    # 2. 启动
    python run_agent.py

    # 3. 浏览器访问
    http://127.0.0.1:8000

首次启动会花 30~60 秒加载 SDXL 模型。日志里出现 "Agent 预热完成" 后再打开浏览器。
"""
import logging
import sys

from agent import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# 降低 uvicorn / watchfiles 噪音
for noisy in ("uvicorn.access", "watchfiles.main", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    errors = config.validate()
    if errors:
        print("\n配置错误：", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)
        print(
            "\n请复制 .env.agent.example 为 .env.agent 并填入 API Key。\n",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvicorn

    mode_banner = (
        "MOCK 模式（假工具，无需模型 / GPU）"
        if config.AGENT_MCP_MODE == "mock"
        else f"REAL 模式 - MCP: {config.MCP_SERVER_SCRIPT}"
    )
    warmup_hint = (
        "  mock 模式瞬启，浏览器可以立刻打开"
        if config.AGENT_MCP_MODE == "mock"
        else "  首次启动需 30~60 秒加载 SDXL 模型，请等待 \"Agent 预热完成\" 日志后再打开浏览器"
    )
    print(
        f"\n  ChromaManga Agent\n"
        f"  Mode: {mode_banner}\n"
        f"  LLM : {config.LLM_MODEL} @ {config.LLM_BASE_URL}\n"
        f"  UI  : http://{config.AGENT_HOST}:{config.AGENT_PORT}\n\n"
        f"{warmup_hint}\n"
    )

    uvicorn.run(
        "agent.server:app",
        host=config.AGENT_HOST,
        port=config.AGENT_PORT,
        log_level="info",
        reload=False,
        ws_ping_interval=30,
        ws_ping_timeout=30,
    )


if __name__ == "__main__":
    main()
