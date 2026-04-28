"""Agent 侧配置：LLM 凭据 + MCP 启动命令 + 文件路径。

所有配置来自环境变量；如果项目根目录存在 `.env.agent`，启动时自动加载。
"""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_env_file = PROJECT_ROOT / ".env.agent"
if _env_file.exists() and load_dotenv is not None:
    load_dotenv(_env_file)

# ── LLM ────────────────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# ── MCP Server ─────────────────────────────────────────────────────────
MCP_SERVER_SCRIPT = str(PROJECT_ROOT / "mcp_server" / "chromamanga_mcp_server.py")
MCP_PYTHON = os.getenv("MCP_PYTHON") or sys.executable

# ── File upload ────────────────────────────────────────────────────────
# 与 core/config.py::inputs_dir 对齐，确保 MCP Server 能读到上传的图
UPLOAD_DIR = PROJECT_ROOT / "data" / "inputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ── Agent runtime ──────────────────────────────────────────────────────
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "50"))
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8000"))

# real: 启动真 MCP 子进程（需模型文件和 GPU）
# mock: 使用 agent/mock_tools.py 里的假工具（开发 / UI 调试用）
AGENT_MCP_MODE = os.getenv("AGENT_MCP_MODE", "real").lower()
if AGENT_MCP_MODE not in ("real", "mock"):
    AGENT_MCP_MODE = "real"


def validate() -> list[str]:
    """启动前体检。返回错误列表，为空表示通过。"""
    errors: list[str] = []
    if not LLM_API_KEY:
        errors.append(
            "LLM_API_KEY 未配置。请复制 .env.agent.example 为 .env.agent 并填入 API Key。"
        )
    if AGENT_MCP_MODE == "real" and not Path(MCP_SERVER_SCRIPT).exists():
        errors.append(f"找不到 MCP server 脚本: {MCP_SERVER_SCRIPT}")
    if not FRONTEND_DIR.exists():
        errors.append(f"找不到前端目录: {FRONTEND_DIR}")
    return errors
