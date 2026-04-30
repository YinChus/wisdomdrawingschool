"""
🌐 智绘书院 MCP Server —— 标准 Anthropic Model Context Protocol 实现

把 academy_agents.py 里的 5 个 @mcp_tool 工具 + RAG 检索能力，
通过官方 MCP SDK 暴露成标准协议 server，任何支持 MCP 的客户端都能调用：

  • Claude Desktop  → stdio 模式
  • Cursor          → stdio 模式
  • VS Code Copilot → stdio 模式
  • 远程 / Web      → SSE 模式
  • 自家 LangGraph  → 通过 mcp_client.py 内嵌调用

启动方式：
  python mcp_server.py           # stdio（默认，给本地 AI 客户端）
  python mcp_server.py --sse     # SSE，监听 http://127.0.0.1:8765/sse
  python mcp_server.py --port 9000 --sse

向客户端注册（Claude Desktop 配置示例）：
  {
    "mcpServers": {
      "academy": {
        "command": "python",
        "args": ["F:/dataset-annotator/mcp_server.py"],
        "cwd": "F:/dataset-annotator"
      }
    }
  }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

# 复用现有实现，零重复
from academy_agents import (
    mcp_calculate,
    mcp_classify_math,
    mcp_get_style,
    mcp_generate_geogebra_prompt,
    rag_search as _rag_search,
    list_tools as _list_internal_tools,
)

# ─── 创建 FastMCP 实例 ──────────────────────────────────────
mcp = FastMCP(
    name="academy-math-tools",
    instructions=(
        "智绘书院数学工具集：\n"
        "• calculate: 数学表达式求值\n"
        "• classify_math: 把题目分类到 12 种高中数学场景\n"
        "• get_style: 获取类别对应的可视化样式（颜色/GeoGebra 配色）\n"
        "• generate_geogebra_prompt: 为题目生成 GeoGebra 代码生成提示词\n"
        "• rag_search: 在指定试卷的题库中做语义检索"
    ),
)


# ─── 工具 1：数学表达式计算器 ──────────────────────────────
@mcp.tool(description="计算数学表达式，例如 sin(pi/3) 或 2**10。返回字符串形式的结果。")
def calculate(expression: str) -> str:
    return mcp_calculate(expression)


# ─── 工具 2：题目分类 ───────────────────────────────────────
@mcp.tool(description=(
    "将一道高中数学题目分类到 12 种场景之一："
    "function / inequality / complex_number / triangle_solution / sequence / "
    "probability / solid_geometry / conic_curve / derivative / trigonometric / "
    "vector / set_logic / other。"
))
def classify_math(content: str) -> str:
    return mcp_classify_math(content)


# ─── 工具 3：获取样式配置 ───────────────────────────────────
@mcp.tool(description=(
    "获取数学类别对应的样式配置（颜色 / 背景 / GeoGebra 主色调），"
    "返回 JSON 字符串。category 应为 classify_math 返回的类别。"
))
def get_style(category: str) -> str:
    return mcp_get_style(category)


# ─── 工具 4：GeoGebra 提示词生成 ────────────────────────────
@mcp.tool(description=(
    "为一道数学题生成 GeoGebra 代码生成提示词（含样式注入）。"
    "参数：category（题目类别）、content（题干）、solution（解析，可选）。"
    "返回拼装好的 prompt 文本，下游 LLM 据此生成 GeoGebra 命令序列。"
))
def generate_geogebra_prompt(category: str, content: str, solution: str = "") -> str:
    return mcp_generate_geogebra_prompt(category=category, content=content, solution=solution)


# ─── 工具 5：RAG 语义检索 ───────────────────────────────────
@mcp.tool(description=(
    "在指定试卷的题库索引中做语义检索，返回 top_k 条最相关片段。"
    "参数：query（检索词）、exam_id（试卷 ID）、top_k（默认 5）。"
    "返回 JSON 数组，每项含 {content, type, question_id, score}。"
))
async def rag_search(query: str, exam_id: str, top_k: int = 5) -> str:
    results = await _rag_search(query=query, exam_id=exam_id, top_k=top_k)
    return json.dumps(results, ensure_ascii=False)


# ─── 资源（Resource）：试卷列表 ─────────────────────────────
@mcp.resource("academy://exams")
def list_exams_resource() -> str:
    """暴露试卷清单作为只读资源，AI 客户端可直接读取。"""
    try:
        from pathlib import Path
        path = Path(__file__).with_name("data") / "academy_exams.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = [
                {"id": e.get("id"), "title": e.get("title"),
                 "question_count": len(e.get("questions", []))}
                for e in data
            ]
            return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"读取失败: {e}"
    return "[]"


# ─── 提示词模板（Prompt）：变式生成 ─────────────────────────
@mcp.prompt(description="生成数学题变式的标准提示词模板，用于 Generator agent")
def variant_generation_prompt(question_content: str, strategy: str = "numbers") -> str:
    """
    可复用的 Prompt 模板。strategy ∈ {numbers, context, figure, extend}
    """
    strategies = {
        "numbers": "保持题型不变，只改换数字和参数",
        "context": "保留核心数学结构，更换实际生活背景",
        "figure": "改换几何图形（如圆柱→圆锥）但保留同类计算",
        "extend": "在原题基础上增加一问，提升综合度",
    }
    desc = strategies.get(strategy, strategies["numbers"])
    return (
        f"请基于以下原题生成一道变式题，策略：{desc}\n\n"
        f"原题：{question_content}\n\n"
        f"输出 JSON：{{\"content\":\"题干\",\"answer\":\"答案\",\"solution\":\"解析\"}}"
    )


# ─── 启动入口 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="智绘书院 MCP Server")
    parser.add_argument("--sse", action="store_true", help="启用 SSE 远程模式（默认 stdio）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # 提前打印能力清单到 stderr，方便调试（不污染 stdio 协议流）
    print(f"[mcp_server] 启动模式: {'SSE' if args.sse else 'stdio'}", file=sys.stderr)
    print(f"[mcp_server] 已注册工具: {[t['name'] for t in _list_internal_tools()]}", file=sys.stderr)

    if args.sse:
        # SSE 模式：远程客户端通过 HTTP 连接
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"[mcp_server] SSE endpoint: http://{args.host}:{args.port}/sse", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        # stdio 模式：本地 AI 客户端通过子进程拉起
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
