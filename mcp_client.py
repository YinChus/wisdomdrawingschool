"""
🔌 多厂商 LLM ↔ MCP 适配器

把 MCP 工具描述自动翻译成各家 LLM 的 Function Calling schema，
并提供统一调用入口 ToolRouter.call_tool(name, **kwargs)。

支持的厂商：
  • OpenAI         (tools=[{type:"function", function:{...}}])
  • Anthropic      (tools=[{name, description, input_schema}])
  • DeepSeek       (兼容 OpenAI schema)
  • 通义千问 Qwen  (兼容 OpenAI schema, dashscope)
  • 智谱 GLM       (兼容 OpenAI schema)
  • Google Gemini  (tools=[{functionDeclarations:[...]}])

三种调用模式（环境变量 MCP_MODE 切换）：
  • inproc  ← 默认。直接调用 academy_agents 里的 Python 函数（最快，零序列化）
  • stdio   ← 通过 subprocess 拉起 mcp_server.py，走 JSON-RPC over stdio
  • sse     ← 连接远程 MCP server（SSE 端点）

用法：
    from mcp_client import ToolRouter
    router = ToolRouter()                       # 默认 inproc
    schemas = router.openai_schema()            # 拿到 OpenAI 格式 tools 列表
    result  = await router.call_tool("classify_math", content="求函数定义域")
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any, Callable, Optional


# ════════════════════════════════════════════════════════════
#  工具元数据（单一真相源）
# ════════════════════════════════════════════════════════════

_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "calculate",
        "description": "计算数学表达式，例如 sin(pi/3) 或 2**10。返回字符串形式的结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python 风格的数学表达式"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "classify_math",
        "description": (
            "将一道高中数学题分类到 12 种场景之一：function / inequality / "
            "complex_number / triangle_solution / sequence / probability / "
            "solid_geometry / conic_curve / derivative / trigonometric / vector / "
            "set_logic / other。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "题目文本"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_style",
        "description": "获取数学类别对应的样式配置（颜色 / 背景 / GeoGebra 主色调），返回 JSON 字符串。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "classify_math 返回的类别"},
            },
            "required": ["category"],
        },
    },
    {
        "name": "generate_geogebra_prompt",
        "description": "为一道数学题生成 GeoGebra 代码生成提示词（含样式注入）。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "题目类别"},
                "content":  {"type": "string", "description": "题干"},
                "solution": {"type": "string", "description": "解析（可选）", "default": ""},
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "rag_search",
        "description": "在指定试卷的题库索引中做语义检索，返回 top_k 条最相关片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "检索词"},
                "exam_id": {"type": "string", "description": "试卷 ID"},
                "top_k":   {"type": "integer", "description": "返回条数", "default": 5},
            },
            "required": ["query", "exam_id"],
        },
    },
]


# ════════════════════════════════════════════════════════════
#  Schema 适配器（一份元数据 → N 家 LLM 格式）
# ════════════════════════════════════════════════════════════

def to_openai_schema(specs: list[dict] = None) -> list[dict]:
    """OpenAI / DeepSeek / Qwen / GLM / Moonshot 通用格式"""
    specs = specs or _TOOL_SPECS
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in specs
    ]


def to_anthropic_schema(specs: list[dict] = None) -> list[dict]:
    """Anthropic Claude 原生格式"""
    specs = specs or _TOOL_SPECS
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in specs
    ]


def to_gemini_schema(specs: list[dict] = None) -> list[dict]:
    """Google Gemini functionDeclarations 格式"""
    specs = specs or _TOOL_SPECS

    def _strip_default(props: dict) -> dict:
        # Gemini 不支持 default 字段，需剔除
        out = {}
        for k, v in props.items():
            if isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items() if kk != "default"}
            else:
                out[k] = v
        return out

    decls = []
    for s in specs:
        params = dict(s["parameters"])
        if "properties" in params:
            params = {**params, "properties": _strip_default(params["properties"])}
        decls.append({
            "name": s["name"],
            "description": s["description"],
            "parameters": params,
        })
    return [{"functionDeclarations": decls}]


# ════════════════════════════════════════════════════════════
#  路由器：根据 MCP_MODE 决定调用方式
# ════════════════════════════════════════════════════════════

class ToolRouter:
    """统一工具调用入口，对上层 Agent 屏蔽底层 transport"""

    def __init__(self, mode: Optional[str] = None, sse_url: Optional[str] = None):
        self.mode = mode or os.environ.get("MCP_MODE", "inproc")
        self.sse_url = sse_url or os.environ.get("MCP_SSE_URL", "http://127.0.0.1:8765/sse")
        self._inproc_funcs: dict[str, Callable] = {}
        self._mcp_session = None  # 懒加载

        if self.mode == "inproc":
            self._init_inproc()

    def _init_inproc(self) -> None:
        """直接绑定到 academy_agents 里的 Python 函数"""
        from academy_agents import (
            mcp_calculate, mcp_classify_math, mcp_get_style,
            mcp_generate_geogebra_prompt, rag_search,
        )
        self._inproc_funcs = {
            "calculate":                  mcp_calculate,
            "classify_math":              mcp_classify_math,
            "get_style":                  mcp_get_style,
            "generate_geogebra_prompt":   mcp_generate_geogebra_prompt,
            "rag_search":                 rag_search,  # async
        }

    # ─── Schema 出口（供上层 LLM SDK 用）────────────────────
    def openai_schema(self) -> list[dict]:
        return to_openai_schema()

    def anthropic_schema(self) -> list[dict]:
        return to_anthropic_schema()

    def gemini_schema(self) -> list[dict]:
        return to_gemini_schema()

    def list_tools(self) -> list[dict]:
        return [{"name": s["name"], "description": s["description"]} for s in _TOOL_SPECS]

    # ─── 统一调用入口 ───────────────────────────────────────
    async def call_tool(self, name: str, **kwargs) -> Any:
        if self.mode == "inproc":
            return await self._call_inproc(name, **kwargs)
        elif self.mode == "stdio":
            return await self._call_mcp_stdio(name, **kwargs)
        elif self.mode == "sse":
            return await self._call_mcp_sse(name, **kwargs)
        else:
            raise ValueError(f"未知 MCP_MODE: {self.mode}")

    async def _call_inproc(self, name: str, **kwargs) -> Any:
        fn = self._inproc_funcs.get(name)
        if not fn:
            return f"工具 {name!r} 不存在"
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return fn(**kwargs)

    async def _ensure_mcp_session(self):
        """懒加载 MCP 客户端会话（stdio / sse）"""
        if self._mcp_session is not None:
            return self._mcp_session

        from mcp import ClientSession
        if self.mode == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],
            )
            self._stdio_cm = stdio_client(params)
            read, write = await self._stdio_cm.__aenter__()
        elif self.mode == "sse":
            from mcp.client.sse import sse_client
            self._stdio_cm = sse_client(self.sse_url)
            read, write = await self._stdio_cm.__aenter__()
        else:
            raise ValueError(self.mode)

        self._session_cm = ClientSession(read, write)
        self._mcp_session = await self._session_cm.__aenter__()
        await self._mcp_session.initialize()
        return self._mcp_session

    async def _call_mcp_stdio(self, name: str, **kwargs) -> Any:
        sess = await self._ensure_mcp_session()
        result = await sess.call_tool(name, kwargs)
        # MCP 返回 list[TextContent]，提取文本
        if hasattr(result, "content") and result.content:
            parts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(parts) if parts else str(result)
        return str(result)

    async def _call_mcp_sse(self, name: str, **kwargs) -> Any:
        return await self._call_mcp_stdio(name, **kwargs)  # 同样的会话接口

    async def aclose(self):
        if self._mcp_session is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._mcp_session = None


# ════════════════════════════════════════════════════════════
#  便捷单例
# ════════════════════════════════════════════════════════════

_default_router: Optional[ToolRouter] = None

def get_router() -> ToolRouter:
    global _default_router
    if _default_router is None:
        _default_router = ToolRouter()
    return _default_router


# ════════════════════════════════════════════════════════════
#  自检：python mcp_client.py
# ════════════════════════════════════════════════════════════

async def _selftest():
    router = get_router()
    print(f"模式: {router.mode}")
    print(f"工具列表: {[t['name'] for t in router.list_tools()]}")

    print("\n[OpenAI schema 预览]")
    print(json.dumps(router.openai_schema()[1], ensure_ascii=False, indent=2))
    print("\n[Anthropic schema 预览]")
    print(json.dumps(router.anthropic_schema()[1], ensure_ascii=False, indent=2))
    print("\n[Gemini schema 预览（前 1 个）]")
    print(json.dumps(router.gemini_schema()[0]["functionDeclarations"][1], ensure_ascii=False, indent=2))

    print("\n[实际调用]")
    r1 = await router.call_tool("classify_math", content="已知圆柱的高为1，球的直径为2")
    print(f"  classify_math → {r1}")
    r2 = await router.call_tool("calculate", expression="sin(pi/3)**2 + cos(pi/3)**2")
    print(f"  calculate → {r2}")
    r3 = await router.call_tool("get_style", category="solid_geometry")
    print(f"  get_style → {r3}")

    await router.aclose()


if __name__ == "__main__":
    asyncio.run(_selftest())
