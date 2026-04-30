"""
智绘书院 — LangGraph 多 Agent 分析引擎 + MCP 工具注册 + RAG 检索

LangGraph 五阶段流水线：
  parse → classify → solve → visualize → validate

MCP 工具：calculate / classify_math / get_style / generate_geogebra_prompt / rag_search

RAG：将题目 + 解析生成 embedding，存储后供学生问答检索。
LangSmith：设置 LANGCHAIN_TRACING_V2=true 自动上报（读取 .env 中的配置）。
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import timezone, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

# ─── 加载 .env.local 环境变量（兼容 project_me 配置）──────────

def _load_env():
    for candidate in [
        Path("F:/dataset-annotator/project_me/projects/.env.local"),
        Path("F:/dataset-annotator/.env"),
        Path(".env.local"),
        Path(".env"),
    ]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
            break

_load_env()

# ─── LangSmith 追踪（读取已加载的环境变量）──────────────────────
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT 由 .env.local 提供
# 修正常见配置错误：endpoint 必须是 api.smith.langchain.com（带 api 前缀）。
# 若用户配成了 https://smith.langchain.com 会导致 405 Not Allowed 噪声日志。
def _sanitize_langsmith_env():
    bad_hosts = ("https://smith.langchain.com", "http://smith.langchain.com",
                 "smith.langchain.com")
    for key in ("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
        val = os.environ.get(key, "").strip().rstrip("/")
        if val and val in bad_hosts:
            os.environ[key] = "https://api.smith.langchain.com"
    # 若启用了 tracing 但缺少 API key，则禁用，避免无意义重试
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true" \
            and not os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

_sanitize_langsmith_env()

# 静音 LangSmith 后台批量上报失败的噪声日志（不影响主功能）
try:
    import logging as _logging
    for _name in ("langsmith.client", "langsmith._internal._background_thread", "langsmith"):
        _logging.getLogger(_name).setLevel(_logging.ERROR + 10)  # 高于 ERROR，等于关闭
except Exception:
    pass

# ─── 数学类别样式色板 ─────────────────────────────────────────

MATH_STYLE: dict[str, dict] = {
    "function":          {"color": "#3b82f6", "bg": "#dbeafe", "ggb": "Blue",       "desmos": "#3b82f6"},
    "derivative":        {"color": "#8b5cf6", "bg": "#ede9fe", "ggb": "Purple",     "desmos": "#8b5cf6"},
    "solid_geometry":    {"color": "#10b981", "bg": "#d1fae5", "ggb": "Green",      "desmos": "#10b981"},
    "conic_curve":       {"color": "#f59e0b", "bg": "#fef3c7", "ggb": "Orange",     "desmos": "#f59e0b"},
    "trigonometric":     {"color": "#ef4444", "bg": "#fee2e2", "ggb": "Red",        "desmos": "#ef4444"},
    "vector":            {"color": "#ec4899", "bg": "#fce7f3", "ggb": "Pink",       "desmos": "#ec4899"},
    "inequality":        {"color": "#0ea5e9", "bg": "#e0f2fe", "ggb": "Cyan",       "desmos": "#0ea5e9"},
    "sequence":          {"color": "#14b8a6", "bg": "#ccfbf1", "ggb": "Teal",       "desmos": "#14b8a6"},
    "probability":       {"color": "#f97316", "bg": "#ffedd5", "ggb": "Orange",     "desmos": "#f97316"},
    "complex_number":    {"color": "#6366f1", "bg": "#e0e7ff", "ggb": "Violet",     "desmos": "#6366f1"},
    "triangle_solution": {"color": "#84cc16", "bg": "#ecfccb", "ggb": "LightGreen", "desmos": "#84cc16"},
    "set_logic":         {"color": "#64748b", "bg": "#f1f5f9", "ggb": "Gray",       "desmos": "#64748b"},
    "other":             {"color": "#64748b", "bg": "#f1f5f9", "ggb": "Gray",       "desmos": "#64748b"},
}

# 六类高视觉化数学题型
VISUAL_CATEGORIES = {
    "solid_geometry", "conic_curve", "function",
    "derivative", "trigonometric", "vector"
}

# ─── LLM 工厂（与原标注系统共享配置）────────────────────────────

def _load_cfg() -> dict:
    cfg_file = Path("F:/dataset-annotator/data/academy_config.json")
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_llm(vision: bool = False, streaming: bool = False):
    """返回 ChatOpenAI 实例，配置共享自标注系统"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("pip install langchain-openai")

    cfg = _load_cfg()
    api_key  = os.getenv("LLM_API_KEY",      cfg.get("api_key", ""))
    base_url = os.getenv("LLM_BASE_URL",      cfg.get("base_url", "https://api.openai.com/v1"))
    if vision:
        model = os.getenv("LLM_VISION_MODEL", cfg.get("vision_model", "gpt-4o"))
    else:
        model = os.getenv("LLM_TEXT_MODEL",   cfg.get("text_model",  "gpt-4"))

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
        streaming=streaming,
    )


# ─── MCP 工具注册表 ───────────────────────────────────────────

_MCP_TOOLS: dict[str, dict] = {}


def mcp_tool(name: str, description: str):
    """装饰器：将函数注册为 MCP 工具"""
    def decorator(fn):
        _MCP_TOOLS[name] = {"name": name, "description": description, "fn": fn}
        return fn
    return decorator


def call_mcp(name: str, **kwargs) -> Any:
    if name not in _MCP_TOOLS:
        return f"MCP 工具 {name!r} 不存在"
    return _MCP_TOOLS[name]["fn"](**kwargs)


def list_tools() -> list[dict]:
    return [{"name": v["name"], "description": v["description"]} for v in _MCP_TOOLS.values()]


@mcp_tool("calculate", "计算数学表达式，例如 sin(pi/3) 或 2**10")
def mcp_calculate(expression: str) -> str:
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@mcp_tool("classify_math", "将数学题目内容分类到 12 种数学场景之一")
def mcp_classify_math(content: str) -> str:
    rules = [
        (r"函数|f\s*\(|y\s*=|定义域|值域|单调|奇偶", "function"),
        (r"不等式|解不等|恒成立", "inequality"),
        (r"复数|实部|虚部|模长|辐角", "complex_number"),
        (r"三角形|正弦定理|余弦定理|面积.*角|外接圆", "triangle_solution"),
        (r"数列|等差|等比|通项|前\s*n\s*项", "sequence"),
        (r"概率|期望|方差|标准差|随机变量|古典概型", "probability"),
        (r"棱柱|棱锥|球|体积|表面积|二面角|空间|正多面体", "solid_geometry"),
        (r"椭圆|双曲线|抛物线|焦点|准线|离心率", "conic_curve"),
        (r"导数|求导|极值|最值.*f|切线斜率|f'\(", "derivative"),
        (r"sin\b|cos\b|tan\b|正弦|余弦|三角函数|弧度", "trigonometric"),
        (r"向量|内积|点积|叉积|共线|平行四边形", "vector"),
        (r"集合|并集|交集|补集|逻辑|命题|充分必要", "set_logic"),
    ]
    for pattern, cat in rules:
        if re.search(pattern, content, re.IGNORECASE):
            return cat
    return "other"


@mcp_tool("get_style", "获取数学类别对应的样式配置（颜色/背景/GeoGebra颜色）")
def mcp_get_style(category: str) -> str:
    return json.dumps(MATH_STYLE.get(category, MATH_STYLE["other"]), ensure_ascii=False)


@mcp_tool("generate_geogebra_prompt", "生成 GeoGebra 代码生成提示词（含样式注入）")
def mcp_generate_geogebra_prompt(category: str, content: str, solution: str = "") -> str:
    style = MATH_STYLE.get(category, MATH_STYLE["other"])
    color = style["ggb"]
    return (
        f"类别：{category}\n"
        f"题目：{content[:400]}\n"
        f"解析：{solution[:300]}\n"
        f"主色调：{color}\n\n"
        "请生成可被 GeoGebra Web 直接 evalCommand 执行的命令序列，每行一条命令，"
        "用注释 `# 步骤N: 描述` 分组。\n\n"
        "⚠️ GeoGebra 语法红线（违反任何一条该命令必失败）：\n"
        "1. **必须显式乘号**：写 `2*p` 不能写 `2p`；写 `3*x` 不能写 `3x`。\n"
        "2. **Curve 必须 5 个参数**：`Curve(<x表达式>, <y表达式>, <参数名>, <下界>, <上界>)`。\n"
        "   例：抛物线 y²=2px 写成 `c=Curve(t^2/(2*p), t, t, -4, 4)`，禁止只写 4 个参数。\n"
        "3. **不要把数字赋给几何对象**：`AB = x(A)` 会得到数字 1 而不是直线。\n"
        "   过 A 平行 y 轴的直线写 `lineAB = Line(A, yAxis)` 或 `Line(A, (0,1))`。\n"
        "4. **变量名只用字母数字下划线**，不要用中文/空格/中划线；点用大写字母 `A,B,C`，"
        "   函数/曲线/直线用小写 `f,g,c,line1`。\n"
        "5. **Intersect 用法**：`Intersect(<对象1>, <对象2>)` 返回首个交点；"
        "   `Intersect(<对象1>, <对象2>, n)` 返回第 n 个；两个参数都必须是几何对象（线/曲线/圆）。\n"
        "6. **避免重定义**：同一个名字只赋值一次，否则后续会出现 `名字_1`。\n"
        "7. **隐函数曲线**用 `ImplicitCurve(<左-右表达式>)`，例如 `c=ImplicitCurve(y^2 - 2*p*x)`。\n"
        "8. **角度**默认弧度，要度数加 `°`，如 `Rotate(A, 30°, O)`。\n"
        "9. **不要使用 Markdown 代码块标记**（``` 之类），直接输出纯文本命令。\n"
        "10. **禁止"双等号方程"**：不能写 `C = y^2 = 2*p*x`。隐函数要用 "
        "`C = ImplicitCurve(y^2 - 2*p*x)`；显函数用 `f(x) = ...`。\n\n"
        "🎯 圆锥曲线（高频踩坑）——**禁止编造 GeoGebra 不存在的签名**：\n"
        "• 抛物线 y²=2px（顶点原点、开口向右）：用 `c=Curve(t^2/(2*p), t, t, -6, 6)`，"
        "  **不要**写 `Parabola((0,0), Vector(1,0), p)`（GeoGebra 没有这个签名）。\n"
        "  GeoGebra 的 `Parabola(<焦点>, <准线>)` 只接收 2 个参数：焦点+准线。\n"
        "• 椭圆 x²/a²+y²/b²=1：用 `Ellipse((-c,0), (c,0), a)`（两焦点+长半轴），"
        "  或 `c=ImplicitCurve(x^2/a^2 + y^2/b^2 - 1)`。\n"
        "• 双曲线：用 `Hyperbola(<焦点1>, <焦点2>, <半实轴>)` 或 `ImplicitCurve(...)`。\n"
        "• 圆：`Circle(<圆心>, <半径>)` 或 `Circle(<点1>, <点2>, <点3>)`。\n\n"
        "🎯 直线相关：\n"
        "• 过点 P 垂直于 x 轴：`Line(P, yAxis)`（不是 `PerpendicularLine(P, Vector(-1,0))`，"
        "  虽然能跑但表意不清；也**不要**用 `x(P)` 这种返回标量的表达式去当直线）。\n"
        "• 过两点的直线：`Line(A, B)`。\n"
        "• 过点 P 平行/垂直某直线 L：`Line(P, L)` / `PerpendicularLine(P, L)`。\n\n"
        "推荐结构：先定义参数（数字赋值），再定义点，再画几何对象，最后做交点/角度等派生。"
    )


# ─── LangGraph 状态定义 ────────────────────────────────────────

class ExamState(TypedDict):
    exam_id: str
    raw_text: str
    file_type: str
    questions: list[dict]
    errors: list[str]
    stage: str
    progress: list[str]   # 实时进度日志


# ─── 五阶段 Agent 节点 ────────────────────────────────────────

async def parse_node(state: ExamState) -> ExamState:
    """阶段1：从原始文本提取结构化题目列表"""
    try:
        from langchain_core.messages import HumanMessage
        llm = get_llm()
        prompt = (
            "你是专业的数学试卷解析器。从以下试卷文本中提取所有题目，以 JSON 数组格式返回。\n\n"
            "每道题格式：{\"number\": 题号, \"type\": \"choice|multi_choice|fill|calculation\", "
            "\"content\": \"完整题目\", \"options\": [\"A...\",\"B...\"] (仅选择题), \"answer\": \"参考答案(如有)\"}\n\n"
            f"试卷文本：\n{state['raw_text'][:8000]}\n\n只返回 JSON 数组。"
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        m = re.search(r"\[[\s\S]*\]", resp.content)
        questions = json.loads(m.group()) if m else []
        progress = state.get("progress", []) + [f"✅ 阶段1解析：提取到 {len(questions)} 道题"]
        return {**state, "questions": questions, "stage": "parsed", "progress": progress}
    except Exception as e:
        return {**state, "stage": "parse_error", "errors": state.get("errors", []) + [f"解析失败: {e}"],
                "progress": state.get("progress", []) + [f"❌ 解析错误: {e}"]}


async def classify_node(state: ExamState) -> ExamState:
    """阶段2：为每道题分类（12种数学场景）"""
    classified = []
    for q in state.get("questions", []):
        cat = call_mcp("classify_math", content=q.get("content", ""))
        style = json.loads(call_mcp("get_style", category=cat))
        classified.append({**q, "category": cat, "style_config": style})
    progress = state.get("progress", []) + [f"✅ 阶段2分类：覆盖 {len(set(q['category'] for q in classified))} 种类别"]
    return {**state, "questions": classified, "stage": "classified", "progress": progress}


async def solve_node(state: ExamState) -> ExamState:
    """阶段3：AI 解题，生成分步解析"""
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    solved = []

    for q in state.get("questions", []):
        if q.get("solution"):
            solved.append(q)
            continue
        cat = q.get("category", "other")
        opts_text = "\n".join(q.get("options") or [])
        prompt = (
            f"你是经验丰富的数学教师，请为以下题目提供详细解题过程。\n"
            f"类别：{cat}\n题目：{q.get('content','')}\n"
            f"{f'选项：{opts_text}' if opts_text else ''}\n\n"
            "请按以下格式：\n[解题思路]\n...\n[解题步骤]\n步骤1: ...\n步骤2: ...\n[答案]\n..."
        )
        try:
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            steps = _parse_steps(resp.content)
            solved.append({**q, "solution": resp.content, "solution_steps": steps})
        except Exception as e:
            solved.append({**q, "solve_error": str(e)})

    progress = state.get("progress", []) + [f"✅ 阶段3解题：解答 {len([q for q in solved if q.get('solution')])} 题"]
    return {**state, "questions": solved, "stage": "solved", "progress": progress}


async def visualize_node(state: ExamState) -> ExamState:
    """阶段4：为六类视觉化题型生成 GeoGebra 代码（含分步快照）"""
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    visualized = []

    for q in state.get("questions", []):
        cat = q.get("category", "other")
        if cat not in VISUAL_CATEGORIES or q.get("geogebra_code"):
            visualized.append(q)
            continue

        prompt_text = call_mcp("generate_geogebra_prompt",
                               category=cat,
                               content=q.get("content", ""),
                               solution=q.get("solution", ""))
        full_prompt = (
            "你是 GeoGebra 专家。根据以下信息生成 GeoGebra 命令序列来可视化该数学题。\n\n"
            f"{prompt_text}\n\n"
            "要求：\n"
            "1. 每行一条 GeoGebra 命令\n"
            "2. 用 # 步骤N: 描述 注释标记每个阶段\n"
            "3. 添加必要标签和颜色\n"
            "4. 确保命令可以在 GeoGebra Classic 中执行\n"
            "只返回命令，不要其他解释。"
        )
        try:
            resp = await llm.ainvoke([HumanMessage(content=full_prompt)])
            code = resp.content.strip()
            snapshots = _parse_snapshots(code)
            visualized.append({**q, "geogebra_code": code, "viz_snapshots": snapshots})
        except Exception as e:
            visualized.append({**q, "viz_error": str(e)})

    count = len([q for q in visualized if q.get("geogebra_code")])
    progress = state.get("progress", []) + [f"✅ 阶段4可视化：生成 {count} 套 GeoGebra 代码"]
    return {**state, "questions": visualized, "stage": "visualized", "progress": progress}


async def validate_node(state: ExamState) -> ExamState:
    """阶段5：验证解题和可视化质量，写入微调数据集"""
    validated = []
    finetune_count = 0

    for q in state.get("questions", []):
        score = 1.0
        issues = []
        if not q.get("content"):
            issues.append("题目内容为空"); score = 0.2
        if q.get("type") == "calculation" and not q.get("solution"):
            issues.append("解答题缺少解析"); score = min(score, 0.5)
        if q.get("category") in VISUAL_CATEGORIES and not q.get("geogebra_code"):
            issues.append("可视化题缺少图形代码"); score = min(score, 0.6)

        if q.get("geogebra_code"):
            finetune_count += 1

        validated.append({
            **q,
            "agent_analysis": {
                "validation_score": round(score, 2),
                "issues": issues,
                "validated": len(issues) == 0,
            },
        })

    avg = sum(q["agent_analysis"]["validation_score"] for q in validated) / max(len(validated), 1)
    progress = state.get("progress", []) + [
        f"✅ 阶段5验证：平均质量 {avg:.2f}，{finetune_count} 条代码已标记入微调数据集"
    ]
    return {**state, "questions": validated, "stage": "completed", "progress": progress}


# ─── 构建 LangGraph ───────────────────────────────────────────

def build_exam_graph():
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages
    except ImportError:
        raise ImportError("pip install langgraph")

    g = StateGraph(ExamState)
    g.add_node("parse",     parse_node)
    g.add_node("classify",  classify_node)
    g.add_node("solve",     solve_node)
    g.add_node("visualize", visualize_node)
    g.add_node("validate",  validate_node)

    g.set_entry_point("parse")
    g.add_edge("parse",     "classify")
    g.add_edge("classify",  "solve")
    g.add_edge("solve",     "visualize")
    g.add_edge("visualize", "validate")
    g.add_edge("validate",  END)

    return g.compile()


_exam_graph = None

def get_exam_graph():
    global _exam_graph
    if _exam_graph is None:
        _exam_graph = build_exam_graph()
    return _exam_graph


# ─── RAG 工具 ─────────────────────────────────────────────────

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """调用 LLM API 的 /embeddings 端点"""
    import httpx

    cfg = _load_cfg()
    api_key  = os.getenv("LLM_API_KEY",  cfg.get("api_key", ""))
    base_url = os.getenv("LLM_BASE_URL", cfg.get("base_url", "https://api.openai.com/v1"))

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "text-embedding-ada-002", "input": texts},
        )
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def rag_search(query: str, exam_id: str, top_k: int = 5) -> list[dict]:
    """从 RagChunk 表中检索与 query 最相似的片段"""
    from academy_db import AsyncSessionLocal, RagChunk
    from sqlalchemy import select

    try:
        q_emb = (await get_embeddings([query]))[0]
    except Exception:
        return []

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(RagChunk).where(RagChunk.exam_id == exam_id)
        )).scalars().all()

    scored = []
    for row in rows:
        if row.embedding:
            scored.append({"content": row.content, "type": row.chunk_type,
                           "question_id": row.question_id,
                           "score": cosine_sim(q_emb, row.embedding)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


async def index_exam_for_rag(exam: dict) -> int:
    """将一份试卷的题目+解析 embedding 写入 RagChunk 表"""
    from academy_db import AsyncSessionLocal, RagChunk

    chunks_data = []
    # 试卷标题
    chunks_data.append({"type": "exam_title", "content": exam.get("title", ""), "qid": None})
    for q in exam.get("questions", []):
        qid = q.get("id")
        if q.get("content"):
            chunks_data.append({"type": "question", "content": q["content"], "qid": qid})
        if q.get("solution"):
            chunks_data.append({"type": "solution", "content": q["solution"], "qid": qid})
        if q.get("answer"):
            chunks_data.append({"type": "answer", "content": q["answer"], "qid": qid})

    if not chunks_data:
        return 0

    texts = [c["content"] for c in chunks_data]
    try:
        embeddings = await get_embeddings(texts)
    except Exception:
        embeddings = [None] * len(texts)

    exam_id = exam.get("id", "")
    async with AsyncSessionLocal() as session:
        for c, emb in zip(chunks_data, embeddings):
            session.add(RagChunk(
                exam_id=exam_id,
                question_id=c["qid"],
                chunk_type=c["type"],
                content=c["content"],
                embedding=emb,
            ))
        await session.commit()

    return len(chunks_data)


# ─── 辅助函数 ─────────────────────────────────────────────────

def _parse_steps(text: str) -> list[dict]:
    steps = []
    current: dict | None = None
    for line in text.split("\n"):
        m = re.match(r"步骤\s*(\d+)[：:]\s*(.*)", line.strip())
        if m:
            if current:
                steps.append(current)
            current = {"step": int(m.group(1)), "content": m.group(2)}
        elif current:
            current["content"] += "\n" + line.strip()
    if current:
        steps.append(current)
    return steps


def _parse_snapshots(code: str) -> list[dict]:
    snapshots = []
    current: dict = {"step": 0, "description": "初始化", "commands": []}
    for line in code.split("\n"):
        m = re.match(r"#\s*步骤\s*(\d+)[：:]?\s*(.*)", line.strip())
        if m:
            if current["commands"]:
                snapshots.append(current)
            current = {"step": int(m.group(1)), "description": m.group(2).strip(), "commands": []}
        elif not line.strip().startswith("#") and line.strip():
            current["commands"].append(line.strip())
    if current["commands"]:
        snapshots.append(current)
    return snapshots
