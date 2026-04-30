"""
🔬 端到端解析性能测量脚本
- 直接调用 server.py 内部函数，不走 HTTP
- 测：本地解析耗时 / LLM 调用耗时 / 答案分发耗时 / 进 LLM 的 token 数 / 题目数
- 对照基线：把整个 PDF base64 + 图片直接投喂多模态大模型 的 token 估算
"""
from __future__ import annotations
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import tiktoken

# 让 pymupdf 渲图也算上对照基线
TARGET_PDFS = [
    Path("data/academy_files/25ffad0e-12b.pdf"),
    Path("data/academy_files/d8a2f341-c6a.pdf"),
]
ANSWER_PDF = Path("data/academy_files/25ffad0e-12b_ans.pdf")

# 选一份存在的
TARGET = next((p for p in TARGET_PDFS if p.exists()), None)
if TARGET is None:
    print("❌ 未找到测试 PDF，请先在网页上传过任一份试卷")
    sys.exit(1)

print(f"📄 目标试卷: {TARGET.name} ({TARGET.stat().st_size/1024:.1f} KB)")

# 导入 server 模块
import server  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
def n_tok(s: str) -> int:
    return len(ENC.encode(s or ""))


async def main():
    cfg = server.read_academy_cfg()
    file_bytes = TARGET.read_bytes()

    # ════ 阶段 1：本地解析 ════
    t0 = time.time()
    extracted = await server._extract_file_content(file_bytes, TARGET.name, cfg)
    t1 = time.time()
    text = extracted.get("content", "") if extracted.get("mode") == "text" else ""
    extract_tokens = n_tok(text)
    print(f"\n[1/3] 📦 本地解析（PyMuPDF）: {t1-t0:.2f}s")
    print(f"      抽取文字: {len(text)} chars / {extract_tokens} tokens")

    # ════ 阶段 2：LLM 结构化解析 ════
    t2 = time.time()
    parsed = await server._llm_parse_exam(extracted, cfg, answer_text="")
    t3 = time.time()
    questions = parsed.get("questions", [])
    print(f"\n[2/3] 🤖 LLM 结构化: {t3-t2:.2f}s")
    print(f"      抽出 {len(questions)} 道题")

    # ════ 阶段 3：答案分发（如有答案文件）════
    distrib_secs = 0
    if ANSWER_PDF.exists():
        ans_bytes = ANSWER_PDF.read_bytes()
        ans_extracted = await server._extract_file_content(ans_bytes, ANSWER_PDF.name, cfg)
        ans_text = ans_extracted.get("content", "")
        exam_record = {"id": "bench", "title": "bench", "questions": questions}
        t4 = time.time()
        dist_summary = await server._distribute_answers_to_questions(exam_record, ans_text)
        t5 = time.time()
        distrib_secs = t5 - t4
        print(f"\n[3/3] 📝 答案分发: {distrib_secs:.2f}s "
              f"(答案文本 {n_tok(ans_text)} tokens, "
              f"分发 {dist_summary.get('distributed',0)}/{len(questions)} 题, "
              f"{'纯正则零 LLM' if distrib_secs < 0.5 else '触发 LLM 兜底'})")
    else:
        print(f"\n[3/3] 📝 答案分发: 跳过（无答案文件）")

    total = (t3 - t0) + distrib_secs
    print(f"\n{'='*60}\n✅ 端到端总耗时: {total:.2f}s | 题目数: {len(questions)}")

    # ════ 对照基线：整文件投喂 ════
    # 模拟把 PDF 转成图（每页渲染为 PNG），再 base64 投喂多模态模型
    print(f"\n{'─'*60}\n📊 对照基线（整文件投喂多模态模型，仅 token 估算）：")
    try:
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total_b64_tokens = 0
        total_img_tokens = 0
        page_count = doc.page_count
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=120)
            png = pix.tobytes("png")
            b64 = base64.b64encode(png).decode()
            total_b64_tokens += n_tok(b64)
            # OpenAI vision 计费经验值：1 张 1024x1024 ≈ 765 tokens
            w, h = pix.width, pix.height
            tiles = ((w + 511) // 512) * ((h + 511) // 512)
            total_img_tokens += 85 + 170 * tiles
        doc.close()

        print(f"   PDF 页数: {page_count}")
        print(f"   方案A（base64 直送文本接口）: {total_b64_tokens:,} tokens")
        print(f"   方案B（视觉接口按 tile 计费）: {total_img_tokens:,} tokens")
    except Exception as e:
        print(f"   计算失败: {e}")
        total_b64_tokens = 0
        total_img_tokens = 0

    # ════ 实际方案 vs 对照 ════
    print(f"\n   实际方案 LLM 输入: {extract_tokens:,} tokens（仅本地抽出的纯文本）")
    if total_img_tokens > 0:
        save_pct = (1 - extract_tokens / total_img_tokens) * 100
        print(f"   vs 视觉接口方案: 节省 {save_pct:.1f}%")
    if total_b64_tokens > 0:
        save_pct2 = (1 - extract_tokens / total_b64_tokens) * 100
        print(f"   vs base64 文本方案: 节省 {save_pct2:.1f}%")

    # ════ 结果落盘 ════
    report = {
        "file": TARGET.name,
        "size_kb": round(TARGET.stat().st_size / 1024, 1),
        "stage_extract_secs": round(t1 - t0, 2),
        "stage_llm_secs": round(t3 - t2, 2),
        "stage_distribute_secs": round(distrib_secs, 2),
        "total_secs": round(total, 2),
        "question_count": len(questions),
        "actual_input_tokens": extract_tokens,
        "baseline_b64_tokens": total_b64_tokens,
        "baseline_vision_tokens": total_img_tokens,
        "savings_vs_vision_pct": round((1 - extract_tokens / total_img_tokens) * 100, 1) if total_img_tokens else None,
        "savings_vs_b64_pct": round((1 - extract_tokens / total_b64_tokens) * 100, 1) if total_b64_tokens else None,
    }
    Path("_bench_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 详细报告已写入: _bench_report.json")


if __name__ == "__main__":
    asyncio.run(main())
