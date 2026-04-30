"""
🔬 出题工厂 + 几何画板 真实性能基准
用真实试卷里的题目跑一遍：
  ① 一题生 N 道变式（agent_factory.run_factory_stream）
  ② 单题生成 GeoGebra 几何代码（mcp_generate_geogebra_prompt + LLM）
"""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
def n_tok(s: str) -> int:
    return len(ENC.encode(s or ""))


def pick_questions():
    """从已有试卷中选取代表性题目：1道几何 + 1道函数 + 1道代数"""
    data = json.load(open("data/academy_exams.json", encoding="utf-8"))
    exam = data[1] if len(data) > 1 else data[0]   # 用 12 题那份
    qs = exam.get("questions", [])

    # 优先按 category 选
    by_cat = {}
    for q in qs:
        cat = q.get("category", "other")
        by_cat.setdefault(cat, []).append(q)

    selected = []
    for prefer in ["solid_geometry", "analytic_geometry", "function", "trigonometry", "algebra"]:
        if prefer in by_cat and by_cat[prefer]:
            selected.append(by_cat[prefer][0])
            if len(selected) >= 3:
                break
    if len(selected) < 3:
        for q in qs:
            if q not in selected:
                selected.append(q)
            if len(selected) >= 3:
                break
    return selected[:3]


# ════════════════════════════════════════════════════════════
#  ① 出题工厂基准
# ════════════════════════════════════════════════════════════

async def bench_factory(question: dict, n: int = 3):
    from agent_factory import run_factory_stream
    print(f"\n{'═'*64}\n🏭 出题工厂：一题生 {n} 道变式")
    print(f"   原题 (Q{question.get('number','?')}, {question.get('category','?')}): "
          f"{question.get('content','')[:60]}…")

    t0 = time.time()
    node_times: dict[str, float] = {}
    last_node = None
    last_ts = t0
    notes_log: list[str] = []
    final_variants = []

    async for ev in run_factory_stream(question, n=n):
        now = time.time()
        if ev["type"] == "progress":
            for note in ev.get("notes", []):
                notes_log.append(note)
                # 简单按表情符号识别节点
                if "Planner" in note:
                    cur = "planner"
                elif "Generator" in note:
                    cur = "generator"
                elif "Solver" in note:
                    cur = "solver"
                elif "GeoDrawer" in note or "几何" in note:
                    cur = "geo_drawer"
                elif "Critic" in note or "审稿" in note:
                    cur = "critic"
                elif "Finalize" in note or "完成" in note:
                    cur = "finalize"
                else:
                    cur = last_node or "unknown"
                if last_node and last_node != cur:
                    node_times[last_node] = node_times.get(last_node, 0) + (now - last_ts)
                    last_ts = now
                last_node = cur
            final_variants = ev.get("variants", final_variants)
        elif ev["type"] == "done":
            final_variants = ev.get("variants", final_variants)
            if last_node:
                node_times[last_node] = node_times.get(last_node, 0) + (now - last_ts)
        elif ev["type"] == "error":
            print(f"   ❌ 错误: {ev['message']}")
            return None

    total = time.time() - t0
    print(f"\n   ⏱️  总耗时: {total:.2f}s ｜ 产出 {len(final_variants)} 道变式")
    for k, v in node_times.items():
        print(f"      └─ {k:12s}: {v:.2f}s")

    # 统计 token：原题 + 变式输出
    src_tok = n_tok(question.get("content", "") + question.get("solution", ""))
    out_tok = sum(n_tok(v.get("content", "") + v.get("answer", "") + v.get("solution", "")) for v in final_variants)
    print(f"   📊 原题 token={src_tok}  ｜ 变式输出 token≈{out_tok}")

    # 显示前两道变式
    for i, v in enumerate(final_variants[:2]):
        print(f"   ✓ 变式{i+1} ({v.get('strategy','?')}/{v.get('difficulty','?')}): "
              f"{v.get('content','')[:50]}…")

    return {
        "total_secs": round(total, 2),
        "variant_count": len(final_variants),
        "node_times": {k: round(v, 2) for k, v in node_times.items()},
        "src_tokens": src_tok,
        "output_tokens": out_tok,
        "notes_count": len(notes_log),
    }


# ════════════════════════════════════════════════════════════
#  ② 几何画板生成基准
# ════════════════════════════════════════════════════════════

async def bench_drawing(question: dict):
    from academy_agents import get_llm, mcp_generate_geogebra_prompt, _parse_snapshots
    from langchain_core.messages import HumanMessage

    print(f"\n{'═'*64}\n📐 GeoGebra 代码生成")
    print(f"   题目 (Q{question.get('number','?')}, {question.get('category','?')}): "
          f"{question.get('content','')[:60]}…")

    cat = question.get("category", "function")
    content = question.get("content", "")
    solution = question.get("solution", "")

    # 阶段 1：MCP 工具构造 prompt
    t0 = time.time()
    prompt_text = mcp_generate_geogebra_prompt(category=cat, content=content, solution=solution)
    t1 = time.time()
    full_prompt = (
        "你是 GeoGebra 专家。根据以下信息生成 GeoGebra 命令序列来可视化该数学题。\n\n"
        f"{prompt_text}\n\n"
        "要求：\n"
        "1. 每行一条 GeoGebra 命令\n"
        "2. 用 # 步骤N: 描述 注释标记每个阶段\n"
        "3. 添加必要标签和颜色\n"
        "4. 确保命令可以在 GeoGebra Classic 中执行\n"
        "只返回命令代码，不要其他解释。"
    )
    in_tok = n_tok(full_prompt)

    # 阶段 2：LLM 调用
    llm = get_llm()
    t2 = time.time()
    resp = await llm.ainvoke([HumanMessage(content=full_prompt)])
    code = resp.content.strip()
    t3 = time.time()

    # 阶段 3：解析步骤快照
    snapshots = _parse_snapshots(code)
    t4 = time.time()

    out_tok = n_tok(code)
    cmd_lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]

    print(f"   ⏱️  总耗时: {(t4-t0):.2f}s")
    print(f"      └─ MCP 构造 prompt:  {(t1-t0)*1000:.1f} ms")
    print(f"      └─ LLM 生成:         {(t3-t2):.2f} s")
    print(f"      └─ 步骤快照解析:     {(t4-t3)*1000:.1f} ms")
    print(f"   📊 输入 prompt: {in_tok} tokens ｜ 输出代码: {out_tok} tokens")
    print(f"   📦 生成 {len(cmd_lines)} 条命令、{len(snapshots)} 个步骤快照")
    print(f"   📝 代码预览:\n      " + (code[:200].replace("\n", "\n      ")))

    return {
        "total_secs": round(t4 - t0, 2),
        "mcp_prompt_ms": round((t1 - t0) * 1000, 1),
        "llm_secs": round(t3 - t2, 2),
        "parse_ms": round((t4 - t3) * 1000, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "command_count": len(cmd_lines),
        "snapshot_count": len(snapshots),
    }


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

async def main():
    questions = pick_questions()
    print(f"📋 选中题目数: {len(questions)}")
    for q in questions:
        print(f"   - Q{q.get('number','?')} [{q.get('category','?')}]: {q.get('content','')[:50]}…")

    report = {"factory": [], "drawing": []}

    # ① 出题工厂：选 1 道几何题跑一次
    geo_q = next((q for q in questions if q.get("category", "").endswith("geometry")), questions[0])
    fac_result = await bench_factory(geo_q, n=3)
    if fac_result:
        report["factory"].append({"question_id": geo_q.get("id"), **fac_result})

    # ② 画板生成：3 道题各跑一次
    for q in questions:
        try:
            d = await bench_drawing(q)
            report["drawing"].append({"question_id": q.get("id"), "category": q.get("category"), **d})
        except Exception as e:
            print(f"   ❌ 画板生成失败: {e}")

    # 汇总
    print(f"\n\n{'█'*64}\n📈 汇总")
    if report["factory"]:
        f = report["factory"][0]
        print(f"\n出题工厂（{f['variant_count']} 道变式）：")
        print(f"   端到端耗时: {f['total_secs']}s")
        print(f"   节点耗时分布: {f['node_times']}")
        print(f"   每道变式平均耗时: {f['total_secs']/max(1,f['variant_count']):.2f}s")
    if report["drawing"]:
        secs = [d["total_secs"] for d in report["drawing"]]
        out_toks = [d["output_tokens"] for d in report["drawing"]]
        cmds = [d["command_count"] for d in report["drawing"]]
        print(f"\n画板生成（{len(secs)} 道）：")
        print(f"   单题耗时: 最快 {min(secs)}s / 平均 {sum(secs)/len(secs):.2f}s / 最慢 {max(secs)}s")
        print(f"   输出代码: 平均 {sum(out_toks)//len(out_toks)} tokens / {sum(cmds)//len(cmds)} 行命令")

    Path("_bench_factory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n💾 详细报告: _bench_factory_report.json")


if __name__ == "__main__":
    asyncio.run(main())
