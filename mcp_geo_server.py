"""
🛰️ MCP Geometry Code Server (stdio)

把"几何/3D/TikZ 代码生成"封装成符合 Model Context Protocol 的工具服务。
- 独立进程，通过 stdio 与上游 (LangGraph Agent) 通信
- 4 个工具：gen_geogebra_2d / gen_geogebra_3d / gen_tikz / validate_geogebra
- 支持 `model` 参数（不传则使用 academy.cfg.json 里的 geo_model 或 text_model）
- 完全无副作用，便于将来切换到你**微调过的几何代码模型**

启动（独立调试）：
    conda activate python39
    python mcp_geo_server.py

正常生产里由 mcp_geo_client.py 通过 stdio 自动拉起。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ════════════════════════════════════════════════════════════
#  配置加载（与 server.py 共享 academy.cfg.json）
# ════════════════════════════════════════════════════════════

_BASE_DIR = Path(__file__).parent
_CFG_FILE = _BASE_DIR / "data" / "academy.cfg.json"


def _load_cfg() -> dict:
    if _CFG_FILE.exists():
        try:
            return json.loads(_CFG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "text_model": os.environ.get("LLM_TEXT_MODEL", os.environ.get("LLM_MODEL", "")),
        "geo_model": os.environ.get("LLM_GEO_MODEL", ""),
    }


# ════════════════════════════════════════════════════════════
#  System Prompts（精修版，针对 GeoGebra/TikZ 命令的语法约束）
# ════════════════════════════════════════════════════════════

_SYS_2D = """你是 GeoGebra 2D 命令专家。给定题目，输出**可直接粘进 GeoGebra 输入栏执行**的命令序列。

硬性规则：
1. 每行一条命令，注释行用 # 开头
2. 点用大写英文字母 A,B,C...；曲线/线段/直线用小写英文字母 a,b,c...
3. 禁止使用中文标签（GeoGebra 不支持）
4. 圆锥曲线**优先用方程定义**：x^2/4 + y^2/9 = 1，而不是 Ellipse[]
5. 对每条几何对象，紧接着可加 SetCaption[obj, "中文说明"]（中文写在引号里 OK）
6. 输出长度建议 5~30 行
7. 只输出命令本身，不要 ```围栏，不要任何解释或前后白话

⚠️ 圆锥曲线高频踩坑（必须遵守）：
- **抛物线 y²=2px**：直接用方程 `c: y^2 = 2*p*x`，**不要**用 `Curve(...)`！
  如果非要用参数曲线，**必须 5 个参数**：`Curve( t^2/(2*p), t, t, -5, 5 )`
  ❌ 错：`Curve((y^2)/(2*p), y, -5, 5)`  ← 只有 4 个参数
- **椭圆**：方程 `e: x^2/a^2 + y^2/b^2 = 1`，或 `Ellipse(F1, F2, a)` 三参数
- **双曲线**：方程 `h: x^2/a^2 - y^2/b^2 = 1`，或 `Hyperbola(F1, F2, a)` 三参数
- 含未定义参数（如 p、a、b）时，**必须先用 `p = 2`、`a = 3` 之类赋具体数值**

示例（题目：椭圆 x²/4+y²=1，过焦点 F2 的弦 AB）：
F1=(-Sqrt(3),0)
F2=(Sqrt(3),0)
e: x^2/4 + y^2 = 1
A=(2,0)
B=(-1,Sqrt(3)/2)
s=Segment(A,B)
SetCaption[F2,"右焦点 F_2"]
"""

_SYS_3D = """你是 GeoGebra 3D 命令专家。针对立体几何题，输出可在 GeoGebra 3D 视图中执行的命令。

硬性规则：
1. 每行一条命令，注释用 # 开头
2. 顶点用大写英文字母 A,B,C,D...
3. 常用命令：
   - 多面体：Pyramid[<底面>], Prism[...], Tetrahedron[A,B,C,D], Cube[A,B]
   - 球/锥/柱：Sphere[<center>,r], Cone[<base>,<apex>], Cylinder[<bottom>,<top>,r]
   - 平面：Plane[A,B,C] 或 Plane[<point>,<vector>]
   - 二面角：Angle[<plane1>,<plane2>]
4. 必要时加 SetCaption[obj,"中文说明"]
5. 输出 5~25 行
6. 只输出命令本身，禁止 ```围栏 / 任何解释

示例（题目：正四面体 ABCD 棱长 2）：
A=(0,0,0)
B=(2,0,0)
C=(1,Sqrt(3),0)
D=(1,Sqrt(3)/3,2*Sqrt(6)/3)
t=Tetrahedron[A,B,C,D]
SetCaption[t,"正四面体"]
"""

_SYS_TIKZ = """你是 LaTeX TikZ 专家。给定题目，输出可直接放入 LaTeX 文档的 tikzpicture 代码。

规则：
1. 必须以 \\begin{tikzpicture} 开始，\\end{tikzpicture} 结束
2. 节点中文用 {\\zhcn ...} 包裹（假设上游已加载中文宏包）
3. 标准比例：scale=1
4. 只输出 tikzpicture 代码块，禁止 ```围栏 和任何解释
"""

_VALID_2D_HEAD = re.compile(r"^[A-Za-z_]\w*\s*[:=]")  # F1=...  e:...
_BAD_PATTERNS = (re.compile(r"```"), re.compile(r"^\s*以下是"), re.compile(r"^\s*Here"))

# ────────────────────────────────────────────────────────────
#  GeoGebra 常用命令签名表（参数个数）— 用来做静态校验
#  key: 命令名（小写） → (允许的参数个数集合, 中文写法说明)
# ────────────────────────────────────────────────────────────
_CMD_SIGS_2D: dict[str, tuple[set[int], str]] = {
    # 曲线/线
    "curve":      ({5}, "Curve( x(t), y(t), t, a, b ) — 5 个参数"),
    "line":       ({2, 3}, "Line( P, Q ) 或 Line( P, 方向向量 )"),
    "segment":    ({2, 3}, "Segment( A, B ) 或 Segment( A, 长度 )"),
    "ray":        ({2}, "Ray( 起点, 方向点 )"),
    "vector":     ({1, 2}, "Vector( 终点 ) 或 Vector( 起点, 终点 )"),
    "polygon":    ({2, 3, 4, 5, 6, 7, 8, 9, 10}, "Polygon( A, B, C, ... ) 至少 3 顶点"),
    # 圆锥曲线
    "circle":     ({2, 3}, "Circle( 圆心, 半径 ) 或 Circle( A, B, C )"),
    "ellipse":    ({3}, "Ellipse( 焦点1, 焦点2, 半长轴 )"),
    "hyperbola":  ({3}, "Hyperbola( 焦点1, 焦点2, 半实轴 )"),
    "parabola":   ({2}, "Parabola( 焦点, 准线 )"),
    "conic":      ({5, 6}, "Conic( 5 个点 ) 或方程"),
    # 交点
    "intersect":  ({2, 3, 4}, "Intersect( 对象1, 对象2 [, 索引] )"),
    # 中点/距离/角度
    "midpoint":   ({1, 2}, "Midpoint( A, B ) 或 Midpoint( 线段 )"),
    "distance":   ({2}, "Distance( A, B )"),
    "angle":      ({1, 2, 3}, "Angle( 角对象 ) 或 Angle( B, A, C )"),
    "perpendicular": ({2}, "Perpendicular( 点, 直线 )"),
    "parallel":   ({2}, "Parallel( 点, 直线 )"),
    "tangent":    ({2}, "Tangent( 点, 圆/曲线 )"),
    "centroid":   ({1}, "Centroid( 多边形/三角形 )"),
    # 函数
    "function":   ({3}, "Function( f, a, b )"),
    "rotate":     ({2, 3}, "Rotate( 对象, 角度 [, 中心] )"),
    "translate":  ({2}, "Translate( 对象, 向量 )"),
    "reflect":    ({2}, "Reflect( 对象, 直线/点 )"),
    "setcaption": ({2}, 'SetCaption( 对象, "中文" )'),
}
_CMD_SIGS_3D: dict[str, tuple[set[int], str]] = {
    "pyramid":     ({2, 3, 4, 5}, "Pyramid( 底面, 顶点 ) 或 Pyramid( A,B,C,D )"),
    "prism":       ({2, 3, 4, 5, 6, 7, 8}, "Prism( 底面, 顶点 ) 或 Prism( A,B,C,A' )"),
    "tetrahedron": ({4}, "Tetrahedron( A, B, C, D )"),
    "cube":        ({2, 3}, "Cube( A, B ) 或 Cube( A, B, dir )"),
    "sphere":      ({2}, "Sphere( 中心, 半径 ) 或 Sphere( 中心, 表面点 )"),
    "cone":        ({2, 3}, "Cone( 底面圆, 顶点 ) 或 Cone( 中心, 顶点, 半径 )"),
    "cylinder":    ({3}, "Cylinder( 底面中心, 顶面中心, 半径 )"),
    "plane":       ({1, 3}, "Plane( 多边形 ) 或 Plane( A, B, C )"),
    "polyhedron":  ({2, 3, 4, 5, 6, 7, 8}, "Polyhedron( 顶点列表 )"),
    "intersect":   ({2, 3, 4}, "Intersect( 对象1, 对象2 [, 索引] )"),
    "angle":       ({1, 2, 3}, "Angle( 角对象 ) 或 Angle( 平面1, 平面2 )"),
    "distance":    ({2}, "Distance( A, B )"),
    "midpoint":    ({1, 2}, "Midpoint( A, B )"),
    "segment":     ({2}, "Segment( A, B )"),
    "line":        ({2}, "Line( P, Q )"),
    "vector":      ({1, 2}, "Vector( 终点 ) 或 Vector( 起点, 终点 )"),
    "setcaption":  ({2}, 'SetCaption( 对象, "中文" )'),
}

_CMD_CALL_RE = re.compile(r"\b([A-Z][A-Za-z]*)\s*([\(\[])")


def _check_cmd_signatures(code: str, engine: str) -> list[str]:
    """逐行扫描，按命令签名表检查参数个数；不识别的命令略过。

    用括号深度匹配抓取整个参数列表，避免 Curve((y^2)/(2*p), y, -5, 5) 这种
    含嵌套括号的命令被截断。
    """
    sigs = _CMD_SIGS_3D if engine == "3d" else _CMD_SIGS_2D
    bad: list[str] = []
    for raw in code.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        rhs = re.sub(r"^[A-Za-z_]\w*\s*[:=]\s*", "", ln, count=1)
        for m in _CMD_CALL_RE.finditer(rhs):
            cmd = m.group(1).lower()
            open_ch = m.group(2)
            close_ch = ")" if open_ch == "(" else "]"
            if cmd not in sigs:
                continue
            # 从 open paren 后逐字符扫，按深度匹配到对应闭括号
            i = m.end()
            depth = 1
            args_chars: list[str] = []
            n = 1  # 至少 1 个参数（除非空）
            empty = True
            while i < len(rhs) and depth > 0:
                ch = rhs[i]
                if ch in "([{":
                    depth += 1
                    args_chars.append(ch)
                elif ch in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
                    args_chars.append(ch)
                elif ch == "," and depth == 1:
                    n += 1
                    args_chars.append(ch)
                    empty = False
                else:
                    if not ch.isspace():
                        empty = False
                    args_chars.append(ch)
                i += 1
            if empty:
                n = 0
            allowed, hint = sigs[cmd]
            if n not in allowed:
                bad.append(f"{ln[:60]} → {cmd} 收到 {n} 参数，期望 {sorted(allowed)}；正确写法：{hint}")
    return bad


# ════════════════════════════════════════════════════════════
#  LLM 调用
# ════════════════════════════════════════════════════════════

async def _call_llm(messages: list, model: str, max_tokens: int = 800) -> str:
    cfg = _load_cfg()
    api_key = cfg.get("api_key", "")
    base_url = (cfg.get("base_url", "") or "").rstrip("/")
    if not api_key or not base_url:
        raise RuntimeError("academy.cfg.json 缺少 api_key/base_url")
    if not model:
        model = cfg.get("geo_model") or cfg.get("text_model") or ""
    if not model:
        raise RuntimeError("未指定 model，且 academy.cfg.json 中没有 geo_model/text_model")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _strip_fence(text: str) -> str:
    """剥离 markdown 围栏与开头白话。"""
    t = text.strip()
    t = re.sub(r"^```\w*\s*|\s*```$", "", t, flags=re.S)
    # 去掉模型可能加的"以下是..."一行
    lines = t.splitlines()
    cleaned = []
    started = False
    for ln in lines:
        if not started:
            if any(p.search(ln) for p in _BAD_PATTERNS):
                continue
            if ln.strip() == "":
                continue
            started = True
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


# ════════════════════════════════════════════════════════════
#  工具实现
# ════════════════════════════════════════════════════════════

async def _gen_geo(engine: str, problem: str, answer: str, model: str) -> dict:
    """画手 → 校验员 → 修补员（最多 1 次）。

    本函数等价于一个三角色微多智能体：
      Drawer  : LLM 出码
      Validator: _validate（含命令签名表）
      Fixer   : 把 warnings 回灌给 LLM 让它重写
    """
    sys_prompt = {"2d": _SYS_2D, "3d": _SYS_3D, "tikz": _SYS_TIKZ}[engine]
    user = (
        f"【题目】\n{problem}\n\n"
        f"【参考答案/关键结论】\n{answer or '(无)'}\n\n"
        f"请生成对应的 {engine.upper()} 代码。"
    )
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    raw = await _call_llm(msgs, model=model, max_tokens=1000)
    code = _strip_fence(raw)
    val = _validate(code, engine)
    fixed_round = 0

    # —— 修补员：发现错误 → 把错误反馈给 LLM 让它重写 ——
    if not val["ok"] and val["warnings"]:
        fixed_round = 1
        fix_msgs = msgs + [
            {"role": "assistant", "content": code},
            {"role": "user", "content": (
                "❌ 上面的代码静态校验未通过，下面是逐条错误：\n"
                + "\n".join(f"- {w}" for w in val["warnings"])
                + "\n\n请**仅输出修正后的完整代码**（不要解释、不要围栏），"
                "确保每条命令的参数个数与 GeoGebra 官方签名一致。"
            )},
        ]
        try:
            raw2 = await _call_llm(fix_msgs, model=model, max_tokens=1000)
            code2 = _strip_fence(raw2)
            val2 = _validate(code2, engine)
            # 只有重写后 warnings 数量减少才采纳，避免越改越糟
            if len(val2["warnings"]) < len(val["warnings"]):
                code, val = code2, val2
        except Exception:
            pass  # 修补失败保留原码

    return {
        "engine": engine,
        "code": code,
        "ok": val["ok"],
        "warnings": val["warnings"],
        "fixed_round": fixed_round,
        "model": model or _load_cfg().get("geo_model") or _load_cfg().get("text_model", ""),
    }


def _validate(code: str, engine: str) -> dict:
    """轻量级语法校验（不真正执行 GeoGebra）。

    校验项：
    1. 围栏残留 / 空代码
    2. 命令格式（X=... / Func(...)）
    3. 中文标签泄漏
    4. **命令签名**：按 _CMD_SIGS_2D/3D 表检查参数个数（能抓到 Curve 4 参这种错）
    """
    warnings: list[str] = []
    if not code or len(code) < 5:
        return {"ok": False, "warnings": ["代码为空或过短"]}
    if "```" in code:
        warnings.append("仍含 markdown 围栏")
    if engine in ("2d", "3d"):
        # 必须至少有一行符合 GeoGebra 命令格式
        lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not any(_VALID_2D_HEAD.match(ln) or "(" in ln for ln in lines):
            warnings.append("未发现 GeoGebra 命令格式（X=... 或 X:... 或 Func[...]）")
        # 中文标签检测（除了 SetCaption 引号里）
        for ln in lines:
            no_quoted = re.sub(r'"[^"]*"', "", ln)
            if re.search(r"[\u4e00-\u9fa5]", no_quoted):
                warnings.append(f"含中文标签（GeoGebra 可能不支持）：{ln[:30]}")
                break
        # 命令签名检查
        warnings.extend(_check_cmd_signatures(code, engine))
    elif engine == "tikz":
        if "\\begin{tikzpicture}" not in code or "\\end{tikzpicture}" not in code:
            warnings.append("缺少 tikzpicture 环境")
    return {"ok": len(warnings) == 0, "warnings": warnings}


# ════════════════════════════════════════════════════════════
#  MCP Server
# ════════════════════════════════════════════════════════════

server = Server("geo-code")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="gen_geogebra_2d",
            description=(
                "为平面几何 / 解析几何题生成 GeoGebra 2D 命令序列（每行一条，可直接粘入 GeoGebra 输入栏）。"
                "支持椭圆、双曲线、抛物线、圆、三角形、四边形等。可选 model 参数指定使用微调后的几何代码模型。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "题目正文"},
                    "answer": {"type": "string", "description": "参考答案/关键结论（可空）"},
                    "model": {"type": "string", "description": "可选：覆盖默认 geo_model"},
                },
                "required": ["problem"],
            },
        ),
        Tool(
            name="gen_geogebra_3d",
            description=(
                "为立体几何题生成 GeoGebra 3D 命令序列（多面体、球、锥、柱、二面角、空间向量等）。"
                "可选 model 参数指定使用微调后的几何代码模型。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "answer": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["problem"],
            },
        ),
        Tool(
            name="gen_tikz",
            description="为题目生成 LaTeX TikZ 代码块（用于试卷渲染或导出）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "answer": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["problem"],
            },
        ),
        Tool(
            name="validate_geogebra",
            description="轻量校验 GeoGebra 命令或 TikZ 代码：检查围栏残留、中文标签、命令格式等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "engine": {"type": "string", "enum": ["2d", "3d", "tikz"]},
                },
                "required": ["code", "engine"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        problem = (arguments.get("problem") or "").strip()
        answer = (arguments.get("answer") or "").strip()
        model = (arguments.get("model") or "").strip()
        if name == "gen_geogebra_2d":
            result = await _gen_geo("2d", problem, answer, model)
        elif name == "gen_geogebra_3d":
            result = await _gen_geo("3d", problem, answer, model)
        elif name == "gen_tikz":
            result = await _gen_geo("tikz", problem, answer, model)
        elif name == "validate_geogebra":
            result = _validate(arguments.get("code", ""), arguments.get("engine", "2d"))
        else:
            raise ValueError(f"未知工具: {name}")
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False),
        )]


async def _main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
