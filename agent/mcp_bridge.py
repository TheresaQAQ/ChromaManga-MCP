"""工具加载层：按 `AGENT_MCP_MODE` 选择真 MCP 或 mock 工具集。

- `real`: 通过 stdio 启动 `mcp_server/chromamanga_mcp_server.py` 子进程，
  用 `langchain-mcp-adapters` 把 MCP 工具转成 LangChain Tool。MCP 侧会加载
  SDXL + ControlNet + LoRA，首次耗时 30~60 秒。
- `mock`: 返回 `agent/mock_tools.py` 里手写的 15 个同签名假工具，不需要模型文件，
  适合在开发机上调 UI / Agent 循环逻辑。
"""
import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_client: Any = None  # MultiServerMCPClient | None
_tools: list[Any] | None = None


async def get_tools() -> list[Any]:
    """懒加载并返回工具列表（真 MCP 或 mock）。仅在单事件循环下使用。"""
    global _client, _tools
    if _tools is not None:
        return _tools

    if config.AGENT_MCP_MODE == "mock":
        from .mock_tools import get_mock_tools
        _tools = get_mock_tools()
        logger.warning(
            "AGENT_MCP_MODE=mock，使用假工具（不连真 MCP Server）。"
            "共 %d 个工具: %s",
            len(_tools),
            [t.name for t in _tools],
        )
        return _tools

    # real 模式
    from langchain_mcp_adapters.client import MultiServerMCPClient

    logger.info(
        "启动 MCP Server: %s %s（首次会加载 SDXL 模型，约 30~60 秒）",
        config.MCP_PYTHON,
        config.MCP_SERVER_SCRIPT,
    )
    _client = MultiServerMCPClient(
        {
            "chromamanga": {
                "command": config.MCP_PYTHON,
                "args": [config.MCP_SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    _tools = await _client.get_tools()
    logger.info("MCP 工具加载完成: %s", [t.name for t in _tools])
    return _tools


def get_client():
    return _client
