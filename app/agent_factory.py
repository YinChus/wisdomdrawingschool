"""
🏭 出题工厂 — 基于 LangGraph 的 Multi-Agent 变式题生成系统

架构图：
                                ┌─────────────┐
                                │   Planner   │  规划要出几道、改什么维度
                                └──────┬──────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
         ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
         │  Generator  │        │  Generator  │        │  Generator  │
         │    (变式 1) │        │    (变式 2) │        │    (变式 3) │
         └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       ▼
                                ┌──────────────┐
                                │   Solver     │  并行解出每道变式题答案/解析
                                └──────┬───────┘
                                       ▼
                                ┌──────────────┐
                                │  GeoDrawer   │  对几何/3D 题生成 GeoGebra/TikZ
                                └──────┬───────┘
                                       ▼
                                ┌──────────────┐
                                │   Critic     │  自洽？答案唯一？难度合理？
                                └──┬────────┬──┘
                                   通过      不通过 → 回到 Generator 修正（≤2 轮）
                                   ▼
                                ┌──────────────┐
                                │   Finalize   │  排序 + 输出
                                └──────────────┘

参考文献：
  - Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al. NeurIPS 2023)
  - AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Microsoft 2023)
  - Plan-and-Solve Prompting (Wang et al. ACL 2023)
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    _LG_AVAILABLE = True
except Exception:
    _LG_AVAILABLE = False

# ════════════════════════════════════════════════════════════
#  防死循环常量（双重保险）
# ════════════════════════════════════════════════════════════

MAX_FIX_ROUNDS = 2     # 单道题最多重写次数（refix → critic 闭环）
MAX_RECURSION  = 25    # LangGraph 全局递归上限（兜底，正常应远低于此）

# ════════════════════════════════════════════════════════════
#  State 定义
# ════════════════════════════════════════════════════════════

class FactoryState(TypedDict, total=False):
    # 输入
    src_question: dict          # 原题：{content, answer, solution, geogebra_code, viz_engine, category}
    n: int                      # 期望生成数量
    # 中间
    plan: list[dict]            # [{strategy, focus, difficulty}]
    variants: list[dict]        # [{idx, content, answer, solution, geogebra_code, viz_engine, status, fix_count}]
    # 输出
    notes: list[str]            # 执行轨迹（给前端流式展示）
    error: str

# ════════════════════════════════════════════════════════════
#  System Prompts
# ════════════════════════════════════════════════════════════

_PLANNER_SYS = """你是高中数学命题专家。给定一道原题，你要规划如何出 N 道**变式题**。

变式策略只能从以下四类中选：
- "numbers"   ：只换数字（最低难度）
- "context"   ：换情境/包装（如把"小明买苹果"换成"工厂生产零件"）
- "figure"    ：换图形结构（如三角形换四边形、椭圆换双曲线）— 仅几何/解析几何题可选
- "extend"    ：在原题基础上加一问 / 加难度（最高难度）

输出严格 JSON，不要任何额外文字：
{
  "考点": "一句话概括原题主要考查的知识点",
  "可视化需求": "yes" 或 "no",
  "plan": [
    {"strategy": "numbers", "focus": "把系数 a=2 换成 a=3", "difficulty": "easy"},
    {"strategy": "context", "focus": "把数列问题改成银行复利问题", "difficulty": "medium"},
    {"strategy": "extend", "focus": "在原(2)问基础上加(3)问求最值", "difficulty": "hard"}
  ]
}
"""

_GENERATOR_SYS = """你是高中数学命题专家。请按指定策略对原题做一次变式。

输出严格 JSON：
{
  "content": "新题目正文（含已知条件和问题）",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."]   // 仅当原题是选择题时才输出，否则给 []
}

⚠️ LaTeX 书写规范（必须严格遵守，否则前端渲染会乱码）：
- 行内公式用单美元符号 `$...$`，独立公式用 `$$...$$`
- 文字用 `\\text{...}`，绝不可写成 `\\ext{...}`
- 反斜杠就是单反斜杠（`\\log`、`\\sin`、`\\geq`），**不要**写成双反斜杠 `\\\\`
- 中文直接写在公式外，不要塞进 `\\text{}` 内做整段中文段落
- 集合用 `\\{x \\mid -1 \\le x \\le 1\\}` 这种格式

要求：
- 必须保持考点不变
- 数据要新（不能跟原题一字不差）
- 表述清晰、自洽、可解
- 不要带"答案"或"解答"
"""

_SOLVER_SYS = """你是高中数学解题专家。给定题目，请给出参考答案与详细解析。

输出严格 JSON：
{
  "answer": "最终答案（简洁）",
  "solution": "完整解题步骤，分'第一步/第二步'，公式用 $..$ 包裹，最后用 ∴ 给结论"
}

⚠️ 关键：LaTeX 命令必须在 JSON 字符串中写成**双反斜杠**（否则 \\b\\t\\n\\f 会被 JSON 转义成控制字符导致前端乱码）：
- 写 "\\\\boldsymbol" 而不是 "\\boldsymbol"
- 写 "\\\\text{x}"   而不是 "\\text{x}"
- 写 "\\\\frac{1}{2}" 而不是 "\\frac{1}{2}"
- 写 "\\\\nabla" 而不是 "\\nabla"
- 其他命令同理：\\\\sin \\\\cos \\\\log \\\\leq \\\\cup \\\\cap \\\\mid \\\\Big 等
"""

_CRITIC_SYS = """你是严格的数学题审稿人。审查给定的【变式题 + 答案 + 解析】，判断是否合格。

不合格的常见情况：
- 题目条件不足或互相矛盾
- 解出的答案不唯一 / 与解析步骤不匹配
- 跟原题考点偏离过大 / 完全雷同
- 难度异常（标"easy"却很难，反之亦然）
- 几何题缺少必要图形描述

输出严格 JSON：
{
  "ok": true 或 false,
  "score": 0-10 的整数,
  "issues": ["问题1", "问题2"],   // 通过则给 []
  "fix_hint": "若不通过，一句话告诉 Generator 该怎么改"
}
"""

_GEO_SYS_2D = (
    "你是 GeoGebra 2D 命令专家。给定题目，输出可直接执行的 GeoGebra 命令序列，"
    "每行一条命令，注释行用 # 开头。"
    "用英文字母命名点（A,B,C），不要中文标签。"
    "只输出命令本身，不要 markdown 围栏，不要任何解释。"
)
_GEO_SYS_3D = (
    "你是 GeoGebra 3D 命令专家。给定立体几何题目，输出可直接执行的 3D 命令序列，"
    "如 Sphere、Cone、Cube、Plane、f(x,y)= 等。"
    "每行一条命令，注释用 # 开头，英文字母命名顶点。"
    "只输出命令本身，不要 markdown 围栏，不要任何解释。"
)

# ════════════════════════════════════════════════════════════
#  LLM 调用（懒导入避免循环依赖）
# ════════════════════════════════════════════════════════════

async def _llm_json(system: str, user: str, max_tokens: int = 1500) -> dict:
    """调 LLM 并强制 JSON 解析。"""
    from server import _call_llm  # 懒导入
    raw = await _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        response_json=True,
    )
    # 防御性解析：去掉可能的 ```json 围栏
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
    try:
        return json.loads(txt)
    except Exception:
        # 兜底：抓第一个 {...}
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0))
        raise

async def _llm_text(system: str, user: str, max_tokens: int = 1200) -> str:
    from server import _call_llm
    return (await _call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )).strip()

# ════════════════════════════════════════════════════════════
#  几何需求判定
# ════════════════════════════════════════════════════════════

_GEO_2D_PATTERNS = [r"三角形", r"四边形", r"圆[^锥柱]", r"椭圆", r"双曲线", r"抛物线", r"切线", r"焦点"]
_GEO_3D_PATTERNS = [r"立体", r"四面体", r"正方体", r"棱柱", r"棱锥", r"球体|球面|球的", r"圆锥", r"圆柱", r"曲面", r"二面角", r"空间几何"]

def _detect_geo_engine(content: str, hint_engine: str = "") -> str:
    """返回 '2d' / '3d' / '' (不需要画图)。"""
    if hint_engine in ("2d", "3d"):
        # 复用原题引擎
        return hint_engine
    if hint_engine == "tikz":
        return "tikz"
    for p in _GEO_3D_PATTERNS:
        if re.search(p, content):
            return "3d"
    for p in _GEO_2D_PATTERNS:
        if re.search(p, content):
            return "2d"
    return ""

# ════════════════════════════════════════════════════════════
#  Node 实现
# ════════════════════════════════════════════════════════════

async def node_planner(state: FactoryState) -> FactoryState:
    src = state["src_question"]
    n = state.get("n", 3)
    user = (
        f"【原题内容】\n{src.get('content','')}\n\n"
        f"【参考答案】\n{src.get('answer','(无)')}\n\n"
        f"【已有解析】\n{(src.get('solution','') or '')[:400]}\n\n"
        f"请规划 {n} 道变式题。"
    )
    try:
        plan_obj = await _llm_json(_PLANNER_SYS, user, max_tokens=800)
        plan = plan_obj.get("plan", [])[:n]
        notes = list(state.get("notes", []))
        notes.append(f"📋 Planner: 考点={plan_obj.get('考点','?')}，规划 {len(plan)} 道变式")
        return {**state, "plan": plan, "notes": notes}
    except Exception as e:
        return {**state, "error": f"Planner 失败: {e}", "notes": state.get("notes", []) + [f"❌ Planner 失败: {e}"]}

async def _gen_one(src: dict, p: dict) -> dict:
    """单条变式生成。"""
    user = (
        f"【原题】\n{src.get('content','')}\n\n"
        f"【变式策略】{p.get('strategy')}\n"
        f"【具体改造】{p.get('focus')}\n"
        f"【目标难度】{p.get('difficulty','medium')}\n\n"
        f"原题选项：{src.get('options') or '(非选择题)'}"
    )
    obj = await _llm_json(_GENERATOR_SYS, user, max_tokens=1200)
    return {
        "content": _to_text(obj.get("content", "")),
        "options": obj.get("options", []) if isinstance(obj.get("options"), list) else [],
        "strategy": p.get("strategy"),
        "difficulty": p.get("difficulty"),
        "focus": p.get("focus"),
    }

async def node_generator(state: FactoryState) -> FactoryState:
    src = state["src_question"]
    plan = state.get("plan", [])
    notes = list(state.get("notes", []))
    notes.append(f"🛠️ Generator: 并行生成 {len(plan)} 道变式题…")
    results = await asyncio.gather(*[_gen_one(src, p) for p in plan], return_exceptions=True)
    variants = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            notes.append(f"  ✗ 变式{i+1} 生成失败: {r}")
            continue
        variants.append({
            "idx": i,
            **r,
            "answer": "",
            "solution": "",
            "geogebra_code": "",
            "viz_engine": "",
            "critic": None,
            "fix_count": 0,
            "status": "drafted",
        })
        notes.append(f"  ✓ 变式{i+1}({r.get('strategy')}/{r.get('difficulty')}): {_to_text(r.get('content',''))[:30]}…")
    return {**state, "variants": variants, "notes": notes}

def _to_text(x) -> str:
    """LLM 偶尔会把 answer/solution 返回成 dict/list，这里强制转成可显示字符串。"""
    if x is None:
        return ""
    if isinstance(x, str):
        return _fix_latex(x)
    if isinstance(x, (dict, list)):
        try:
            return _fix_latex(json.dumps(x, ensure_ascii=False))
        except Exception:
            return str(x)
    return str(x)


_LATEX_FIXES = [
    # ── 关键修复：LLM 在 JSON 里写 \boldsymbol \text \nabla \rho 等，
    # JSON 反斜杠转义会把 \b → 退格(U+0008), \t → 制表(U+0009),
    # \n → 换行, \r → 回车, \f → 换页。还原回 \b \t \n \r \f。
    (re.compile("\u0008"), r"\\b"),   # backspace → \b（如 \boldsymbol、\bf 等）
    (re.compile("\u0009"), r"\\t"),   # tab → \t（如 \text、\tan、\tau 等）
    (re.compile("\u000c"), r"\\f"),   # form feed → \f（如 \frac、\forall 等）
    # \n / \r 是合法换行，仅当紧邻 LaTeX 命令字母时才还原
    (re.compile(r"\n(?=[a-zA-Z]{2,}\b)"), r"\\n"),
    (re.compile(r"\r(?=[a-zA-Z]{2,}\b)"), r"\\r"),

    # 错把 \text 写成 \ext（极少数情况下 LLM 会这样输出）
    (re.compile(r"\\ext\{"), r"\\text{"),
    # 错误的 $ \\ \text{} \\ \text{} 残片
    (re.compile(r"\$\s*\\\\\s*\\text\{\}\s*(?:\\\\\s*\\text\{\})*\s*\$"), ""),
    # LLM 双重转义：\\{ → \{   \\} → \}   \\| → \|
    (re.compile(r"\\\\(\{|\}|\||,|;|:)"), r"\\\1"),
    # \\cmd（双反斜杠开头的常见命令）→ \cmd
    (re.compile(r"\\\\(?=(?:Big|big|Bigg|bigg|left|right|frac|sqrt|sum|int|prod|lim|infty|cdot|times|"
                r"leq|geq|neq|in|notin|subset|supset|cup|cap|setminus|"
                r"alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega|"
                r"sin|cos|tan|log|ln|exp|max|min|"
                r"mid|parallel|perp|rightarrow|leftarrow|to|"
                r"mathbb|mathcal|mathrm|mathbf|text|overline|underline|hat|tilde|bar|vec|boldsymbol|"
                r"le|ge|ne|cdots|ldots)\b)"), r"\\"),
    # 残留的 \\\\（成对反斜杠）→ \\ 换行
    (re.compile(r"\\\\\\\\"), r"\\\\"),
]

# 常见 LaTeX 命令；用于"裸命令自动包 $...$"兜底
_LATEX_CMDS = (
    "frac|sqrt|sum|int|prod|lim|infty|cdot|times|div|pm|mp|"
    "leq|geq|neq|approx|equiv|sim|propto|in|notin|subset|supset|cup|cap|setminus|"
    "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|"
    "pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    "Alpha|Beta|Gamma|Delta|Theta|Lambda|Pi|Sigma|Phi|Psi|Omega|"
    "sin|cos|tan|cot|sec|csc|log|ln|exp|max|min|"
    "left|right|big|Big|bigg|Bigg|"
    "mid|parallel|perp|angle|triangle|circ|"
    "rightarrow|leftarrow|Rightarrow|Leftarrow|to|mapsto|"
    "mathbb|mathcal|mathrm|mathbf|text|overline|underline|hat|tilde|bar|vec|"
    "le|ge|ne|cdots|ldots|vdots|ddots"
)
# 匹配"包含至少一个 \cmd 的 ASCII/数学字符段"——以中文或行首/尾边界为切分
_MATH_CHAR = r"[A-Za-z0-9_^{}\\\s\.,;:=+\-*/<>|()\[\]!?'\"]"
_BARE_LATEX_RE = re.compile(
    r"(" + _MATH_CHAR + r"*\\(?:" + _LATEX_CMDS + r")(?![A-Za-z])" + _MATH_CHAR + r"*)"
)

def _autowrap_latex(s: str) -> str:
    """把没被 $...$ 包住的裸 LaTeX 表达式自动加上 $...$，避免前端不渲染。"""
    if not isinstance(s, str) or "\\" not in s:
        return s
    # 1) 按 $...$ 切段，只在非数学段做 wrap
    parts = re.split(r"(\$+[^\$]*\$+)", s)
    out = []
    for seg in parts:
        if seg.startswith("$"):
            out.append(seg); continue
        # 2) 按中文字符切，每段 ASCII 子串里再找裸 LaTeX
        sub_parts = re.split(r"([\u4e00-\u9fff]+|[。，、；！？：])", seg)
        for sp in sub_parts:
            if not sp or re.match(r"[\u4e00-\u9fff。，、；！？：]", sp[:1]):
                out.append(sp); continue
            # 在 ASCII 段内找含 \cmd 的子串并包上 $..$
            m = _BARE_LATEX_RE.search(sp)
            if m:
                wrapped = _BARE_LATEX_RE.sub(lambda mm: "$" + mm.group(1).strip() + "$", sp)
                out.append(wrapped)
            else:
                out.append(sp)
    return "".join(out)

def _fix_latex(s: str) -> str:
    """修补 LLM 偶发的 LaTeX 拼写错误，避免前端 KaTeX 渲染失败。"""
    if not isinstance(s, str):
        return s
    for pat, rep in _LATEX_FIXES:
        s = pat.sub(rep, s)
    s = _autowrap_latex(s)
    return s

async def _solve_one(v: dict) -> dict:
    user = f"【题目】\n{v['content']}\n\n选项: {v.get('options') or '(无)'}"
    obj = await _llm_json(_SOLVER_SYS, user, max_tokens=1500)
    return {**v, "answer": _to_text(obj.get("answer", "")), "solution": _to_text(obj.get("solution", ""))}

async def node_solver(state: FactoryState) -> FactoryState:
    notes = list(state.get("notes", []))
    notes.append(f"🧮 Solver: 并行求解 {len(state.get('variants', []))} 道变式题…")
    results = await asyncio.gather(
        *[_solve_one(v) for v in state.get("variants", [])],
        return_exceptions=True,
    )
    variants = []
    for orig, r in zip(state.get("variants", []), results):
        if isinstance(r, Exception):
            notes.append(f"  ✗ 变式{orig['idx']+1} 求解失败: {r}")
            variants.append({**orig, "answer": "(求解失败)", "solution": str(r)})
        else:
            variants.append(r)
            notes.append(f"  ✓ 变式{orig['idx']+1} 答案: {_to_text(r.get('answer',''))[:40]}")
    return {**state, "variants": variants, "notes": notes}

async def _draw_one(v: dict, src_engine: str) -> dict:
    """走 MCP 协议向 mcp_geo_server 发请求生成几何代码。"""
    engine = _detect_geo_engine(v["content"], src_engine)
    if not engine:
        return {**v, "viz_engine": ""}
    try:
        from mcp_geo_client import gen_geo_code  # 懒导入，避免无 mcp 时阻塞 import
        result = await gen_geo_code(engine, v["content"], v.get("answer", ""))
        if not result.get("code"):
            return {
                **v,
                "viz_engine": engine,
                "geogebra_code": f"# MCP 生成失败: {result.get('error','无返回')}",
                "geo_warnings": result.get("warnings", []),
                "geo_model": result.get("model", ""),
            }
        return {
            **v,
            "viz_engine": engine,
            "geogebra_code": result.get("code", ""),
            "geo_warnings": result.get("warnings", []),
            "geo_fixed_round": result.get("fixed_round", 0),
            "geo_model": result.get("model", ""),
        }
    except Exception as e:
        return {**v, "viz_engine": engine, "geogebra_code": f"# 自动生成失败: {e}"}

async def node_drawer(state: FactoryState) -> FactoryState:
    src_engine = (state["src_question"].get("viz_engine") or "").lower()
    notes = list(state.get("notes", []))
    notes.append(f"🎨 GeoDrawer (MCP): 通过 mcp_geo_server 生成几何代码…")
    results = await asyncio.gather(
        *[_draw_one(v, src_engine) for v in state.get("variants", [])],
        return_exceptions=True,
    )
    variants = []
    for orig, r in zip(state.get("variants", []), results):
        if isinstance(r, Exception):
            variants.append(orig)
            notes.append(f"  ✗ 变式{orig['idx']+1} MCP 调用异常: {r}")
        else:
            variants.append(r)
            if r.get("viz_engine"):
                warns = r.get("geo_warnings") or []
                tag = f" ⚠️{len(warns)}处" if warns else ""
                fixed = r.get("geo_fixed_round", 0)
                fix_tag = f" 🔧自愈{fixed}轮" if fixed else ""
                model_tag = f" [{r.get('geo_model','')}]" if r.get("geo_model") else ""
                notes.append(
                    f"  ✓ 变式{orig['idx']+1} 生成 {r['viz_engine'].upper()} 代码（{len(r.get('geogebra_code',''))} 字符）{tag}{fix_tag}{model_tag}"
                )
            else:
                notes.append(f"  · 变式{orig['idx']+1} 无需画图")
    return {**state, "variants": variants, "notes": notes}

async def _critic_one(v: dict) -> dict:
    user = (
        f"【变式题目】\n{v['content']}\n\n"
        f"【参考答案】{v.get('answer','')}\n\n"
        f"【解析】\n{(v.get('solution','') or '')[:600]}\n\n"
        f"【期望难度】{v.get('difficulty','')}    【变式策略】{v.get('strategy','')}"
    )
    try:
        obj = await _llm_json(_CRITIC_SYS, user, max_tokens=500)
        # 防御性：LLM 返回不作数则强制赋默认
        obj.setdefault("ok", True)
        obj.setdefault("score", 6)
        obj.setdefault("issues", [])
        obj.setdefault("fix_hint", "")
        return {**v, "critic": obj}
    except Exception as e:
        # 审稿本身崩了 → 不返输带入下轮重写（避免无限重试）
        return {**v, "critic": {"ok": True, "score": 5, "issues": [], "fix_hint": "", "_warn": f"Critic 调用失败：{e}"}}

async def node_critic(state: FactoryState) -> FactoryState:
    notes = list(state.get("notes", []))
    notes.append(f"🔍 Critic: 审稿 {len(state.get('variants', []))} 道题…")
    results = await asyncio.gather(*[_critic_one(v) for v in state.get("variants", [])])
    variants = []
    for v in results:
        c = v.get("critic") or {}
        ok = c.get("ok", True)
        score = c.get("score", 0)
        v["status"] = "passed" if ok else "needs_fix"
        variants.append(v)
        flag = "✅" if ok else "⚠️"
        notes.append(f"  {flag} 变式{v['idx']+1} 评分 {score}/10" + (f"，问题: {'; '.join(c.get('issues', []))}" if not ok else ""))
    return {**state, "variants": variants, "notes": notes}

async def _refix_one(v: dict, src: dict) -> dict:
    """根据 Critic 反馈重新生成 + 重新求解。"""
    c = v.get("critic") or {}
    user = (
        f"【原题】\n{src.get('content','')}\n\n"
        f"【上一版变式】\n{v['content']}\n\n"
        f"【审稿意见】{c.get('fix_hint','请改进')}\n"
        f"【原变式策略】{v.get('strategy')}    【目标难度】{v.get('difficulty')}\n\n"
        f"请按审稿意见修改并输出新版本。"
    )
    obj = await _llm_json(_GENERATOR_SYS, user, max_tokens=1200)
    new_v = {
        **v,
        "content": _to_text(obj.get("content", v["content"])),
        "options": obj.get("options", v.get("options", [])) if isinstance(obj.get("options", v.get("options", [])), list) else [],
    }
    # 重新求解
    return await _solve_one(new_v)

async def node_refix(state: FactoryState) -> FactoryState:
    src = state["src_question"]
    notes = list(state.get("notes", []))
    fail = [v for v in state.get("variants", []) if v.get("status") == "needs_fix" and v.get("fix_count", 0) < MAX_FIX_ROUNDS]
    if not fail:
        return state
    notes.append(f"🔁 Refix: 修复 {len(fail)} 道题（第 {fail[0].get('fix_count',0)+1}/{MAX_FIX_ROUNDS} 轮）")
    fixed = await asyncio.gather(*[_refix_one(v, src) for v in fail], return_exceptions=True)
    fixed_map = {}
    for orig, r in zip(fail, fixed):
        if isinstance(r, Exception):
            fixed_map[orig["idx"]] = {**orig, "fix_count": orig.get("fix_count", 0) + 1}
        else:
            fixed_map[orig["idx"]] = {**r, "fix_count": orig.get("fix_count", 0) + 1, "status": "drafted"}
    variants = [fixed_map.get(v["idx"], v) for v in state["variants"]]
    return {**state, "variants": variants, "notes": notes}

def route_after_critic(state: FactoryState) -> str:
    """有任何一道可修复且未耗尽重试次数 → refix；否则 finalize。
    双重保险：与 MAX_FIX_ROUNDS 严格同步，防止死循环。"""
    for v in state.get("variants", []):
        if v.get("status") == "needs_fix" and v.get("fix_count", 0) < MAX_FIX_ROUNDS:
            return "refix"
    return "finalize"

async def node_finalize(state: FactoryState) -> FactoryState:
    notes = list(state.get("notes", []))
    variants = []
    for v in state.get("variants", []):
        # 耗尽重试仍不过 → 标终态 failed_review（供前端黄牌提示）
        if v.get("status") == "needs_fix" and v.get("fix_count", 0) >= MAX_FIX_ROUNDS:
            v = {**v, "status": "failed_review"}
        variants.append(v)
    variants.sort(
        key=lambda v: ("easy", "medium", "hard").index(v.get("difficulty", "medium")) if v.get("difficulty") in ("easy", "medium", "hard") else 1,
    )
    passed = sum(1 for v in variants if v.get("status") == "passed")
    failed = sum(1 for v in variants if v.get("status") == "failed_review")
    notes.append(f"🎯 完成：{passed} 道通过审稿，{failed} 道耗尽重试仍需人工复核")
    return {**state, "variants": variants, "notes": notes}

# ════════════════════════════════════════════════════════════
#  Graph 构建
# ════════════════════════════════════════════════════════════

def build_factory_graph():
    if not _LG_AVAILABLE:
        raise RuntimeError("LangGraph 未安装，请运行: pip install langgraph")
    g = StateGraph(FactoryState)
    g.add_node("planner",   node_planner)
    g.add_node("generator", node_generator)
    g.add_node("solver",    node_solver)
    g.add_node("drawer",    node_drawer)
    g.add_node("critic",    node_critic)
    g.add_node("refix",     node_refix)
    g.add_node("finalize",  node_finalize)

    g.set_entry_point("planner")
    g.add_edge("planner",   "generator")
    g.add_edge("generator", "solver")
    g.add_edge("solver",    "drawer")
    g.add_edge("drawer",    "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "refix":    "refix",
        "finalize": "finalize",
    })
    g.add_edge("refix",   "critic")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())

_compiled = None
def get_factory():
    global _compiled
    if _compiled is None:
        _compiled = build_factory_graph()
    return _compiled

# ════════════════════════════════════════════════════════════
#  对外：流式跑一次（产出 SSE 事件）
# ════════════════════════════════════════════════════════════

async def run_factory_stream(src_question: dict, n: int = 3) -> AsyncIterator[dict]:
    """
    流式生成事件：
      {"type": "node", "node": "planner",   "notes": [...]}
      {"type": "node", "node": "generator", "notes": [...], "variants": [...]}
      ...
      {"type": "done",   "variants": [...]}
      {"type": "error",  "message": "..."}
    """
    if not _LG_AVAILABLE:
        yield {"type": "error", "message": "LangGraph 未安装，请运行: pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple"}
        return

    graph = get_factory()
    cfg = {
        "configurable": {"thread_id": f"fac-{id(src_question)}"},
        "recursion_limit": MAX_RECURSION,   # ← 全局兑底：超过则抛 GraphRecursionError
    }
    init_state: FactoryState = {"src_question": src_question, "n": n, "variants": [], "notes": []}
    last_notes_len = 0
    final_state: Optional[FactoryState] = None

    try:
        async for event in graph.astream(init_state, cfg, stream_mode="values"):
            final_state = event
            notes = event.get("notes", [])
            new_notes = notes[last_notes_len:]
            last_notes_len = len(notes)
            yield {
                "type": "progress",
                "notes": new_notes,
                "variants": event.get("variants", []),
            }
            await asyncio.sleep(0)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[agent_factory] graph crash:\n" + tb, flush=True)
        try:
            from pathlib import Path
            Path(__file__).with_name("data").mkdir(exist_ok=True)
            (Path(__file__).parent / "data" / "factory_last_error.log").write_text(tb, encoding="utf-8")
        except Exception:
            pass
        last_line = next((ln for ln in reversed(tb.splitlines()) if ln.strip() and not ln.startswith(" ")), "")
        yield {"type": "error", "message": f"{type(e).__name__}: {e} | {last_line}"}
        return

    if final_state and final_state.get("error"):
        yield {"type": "error", "message": final_state["error"]}
    else:
        yield {"type": "done", "variants": (final_state or {}).get("variants", [])}
