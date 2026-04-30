"""
🛰️ MCP Geometry Code Client

为 LangGraph Agent 提供"通过 MCP 协议"调用 mcp_geo_server.py 的能力。

设计要点：
- **每次调用 fresh session**：MCP 的 stdio_client 基于 anyio TaskGroup，
  其 cancel scope 必须在同一 task 中 enter/exit。LangGraph 节点会并发跑
  在不同 asyncio task 里，单例 session 会触发 "cancel scope in a different
  task" 异常。所以我们牺牲一点启动开销，每次调用都拉起→使用→关闭一次完整会话。
- **可通过环境变量 MCP_GEO_PYTHON 指定 python.exe**（默认走当前解释器）

为什么用 MCP 而不是直接调函数？
1. 协议解耦：将来 mcp_geo_server.py 可以独立部署到 GPU 节点，或替换为
   团队微调后的"几何代码模型"专用服务，主程序不动。
2. 工具复用：同一个 MCP server 可以被 Claude Desktop / VS Code Copilot /
   评测脚本同时调用，统一 prompt + 校验逻辑。
3. 训练数据来源：每次调用产生 (problem, code, ok, warnings) 五元组，
   天然就是 SFT 数据。
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


_BASE_DIR = Path(__file__).parent
_SERVER_PATH = _BASE_DIR / "mcp_geo_server.py"


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=os.environ.get("MCP_GEO_PYTHON", sys.executable),
        args=[str(_SERVER_PATH)],
        env={**os.environ},
    )


async def _call_once(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """单次会话：拉起 → initialize → call_tool → 关闭。

    所有 enter/exit 都在调用者所在的同一个 asyncio task 内完成，
    彻底规避 anyio 跨 task cancel scope 报错。
    """
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(_server_params()))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        result = await session.call_tool(tool, arguments=arguments)
        if not result.content:
            return {"ok": False, "error": "empty MCP response"}
        text = result.content[0].text  # type: ignore[union-attr]
        try:
            return json.loads(text)
        except Exception:
            return {"ok": False, "error": "MCP returned non-JSON", "raw": text}


async def list_tools() -> list[dict[str, Any]]:
    """调试用：列出 MCP server 暴露的工具。"""
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(_server_params()))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        resp = await session.list_tools()
        return [
            {"name": t.name, "description": t.description, "schema": t.inputSchema}
            for t in resp.tools
        ]


# ════════════════════════════════════════════════════════════
#  便捷封装（给 agent_factory 使用）
# ════════════════════════════════════════════════════════════

async def gen_geo_code(engine: str, problem: str, answer: str = "", model: str = "") -> dict:
    """通过 MCP 生成几何代码。

    返回 {engine, code, ok, warnings, model} 或 {ok:False, error:...}。
    """
    tool = {
        "2d": "gen_geogebra_2d",
        "3d": "gen_geogebra_3d",
        "tikz": "gen_tikz",
    }.get(engine)
    if not tool:
        return {"ok": False, "error": f"不支持的 engine: {engine}"}
    args: dict[str, Any] = {"problem": problem, "answer": answer}
    if model:
        args["model"] = model
    try:
        return await _call_once(tool, args)
    except Exception as e:
        return {"ok": False, "error": f"MCP 调用异常: {e}"}
