"""
智绘书院 — 新功能 API Router
挂载到 server.py 的 FastAPI app 上。

路由列表：
  PUT  /api/academy/exams/{eid}/questions/{qid}           内联编辑题目
  GET  /api/academy/files/{filename}                       本地文件下载
  GET  /api/academy/questions/{qid}/resources              题目资源列表
  POST /api/academy/questions/{qid}/resources              上传资源
  DEL  /api/academy/questions/{qid}/resources/{rid}        删除资源
  GET  /api/academy/questions/{qid}/visualizations         可视化列表
  POST /api/academy/questions/{qid}/visualizations         新增可视化
  POST /api/academy/questions/{qid}/visualizations/generate  AI生成可视化
  PUT  /api/academy/questions/{qid}/visualizations/{vid}   更新可视化
  DEL  /api/academy/questions/{qid}/visualizations/{vid}   删除可视化
  POST /api/academy/student/chat                           开始/续接对话
  GET  /api/academy/student/chat/stream                    流式 SSE 问答
  POST /api/academy/student/feedback                       匿名反馈提交
  GET  /api/academy/teacher/feedback/{exam_id}             查看反馈(教师)
  PUT  /api/academy/teacher/feedback/{fid}/resolve         标记已解决
  POST /api/academy/exams/{eid}/run-agents  (SSE)          五阶段 Agent 流水线
  POST /api/academy/exams/{eid}/index-rag                  RAG 索引
  GET  /api/academy/finetune/export                        微调数据集导出
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ─── 延迟导入（DB / Storage / Agents 在 server 启动后才安全使用）──

academy_router = APIRouter()

BASE_DIR          = Path("F:/dataset-annotator/data")
ACADEMY_DB_FILE   = BASE_DIR / "academy_exams.json"
LOCAL_UPLOAD_DIR  = BASE_DIR / "academy_files"
LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─── 工具函数 ─────────────────────────────────────────────────

def _read_exams() -> list[dict]:
    if not ACADEMY_DB_FILE.exists():
        return []
    text = ACADEMY_DB_FILE.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def _write_exams(data: list[dict]) -> None:
    tmp = ACADEMY_DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, ACADEMY_DB_FILE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── 本地文件服务 ─────────────────────────────────────────────

@academy_router.get("/api/academy/files/{filename:path}")
async def serve_local_file(filename: str):
    """将 academy_storage 保存的本地文件通过 HTTP 提供下载。"""
    # 安全校验：不允许路径穿越
    safe = Path(filename).name  # 去掉任何目录前缀
    if safe != filename.replace("/", "__"):
        safe = filename.replace("/", "__")
    target = LOCAL_UPLOAD_DIR / safe
    if not target.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(str(target))


# ─── 题目内联编辑 ─────────────────────────────────────────────

@academy_router.put("/api/academy/exams/{eid}/questions/{qid}")
async def update_question(eid: str, qid: str, body: dict):
    exams = _read_exams()
    exam = next((e for e in exams if e["id"] == eid), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    qs = exam.get("questions", [])
    target = next((q for q in qs if str(q.get("id", "")) == qid), None)
    if not target:
        raise HTTPException(404, "题目不存在")

    # 允许修改的字段
    tracked = ("content", "answer", "solution", "options", "type", "score",
               "geogebra_code", "desmos_json", "category")
    changed_fields = []
    for field in tracked:
        if field in body and body[field] != target.get(field):
            target[field] = body[field]
            changed_fields.append(field)
    # 其它非追踪字段也允许写入但不会触发"待重新发布"
    for field in body:
        if field not in tracked and field not in {"id", "number"}:
            target[field] = body[field]
    target["updated_at"] = _now()
    # 若试卷已发布且关键字段被修改，则标记该题"待重新发布"
    if changed_fields and exam.get("published") and target.get("published_at"):
        target["has_pending_changes"] = True
        target["pending_fields"] = sorted(set(target.get("pending_fields", []) + changed_fields))
    exam["updated_at"] = _now()
    _write_exams(exams)
    return {"ok": True, "question": target,
            "has_pending_changes": bool(target.get("has_pending_changes"))}


@academy_router.post("/api/academy/exams/{eid}/questions/{qid}/republish")
async def republish_question(eid: str, qid: str):
    """重新发布单题：清除该题的"待重新发布"标记，更新 published_at 与 version。"""
    exams = _read_exams()
    exam = next((e for e in exams if e["id"] == eid), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    if not exam.get("published"):
        raise HTTPException(400, "试卷尚未发布，请先到「发布配置」发布整卷")
    qs = exam.get("questions", [])
    target = next((q for q in qs if str(q.get("id", "")) == qid or str(q.get("number", "")) == qid), None)
    if not target:
        raise HTTPException(404, "题目不存在")
    now = _now()
    target["published_at"] = now
    target["version"] = int(target.get("version", 0)) + 1
    target["has_pending_changes"] = False
    target["pending_fields"] = []
    exam["updated_at"] = now
    _write_exams(exams)
    return {
        "ok": True, "qid": qid,
        "published_at": now,
        "version": target["version"],
    }


@academy_router.post("/api/academy/exams/{eid}/questions/republish-all")
async def republish_all_questions(eid: str):
    """一键重新发布所有"待重新发布"的题。"""
    exams = _read_exams()
    exam = next((e for e in exams if e["id"] == eid), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    if not exam.get("published"):
        raise HTTPException(400, "试卷尚未发布")
    now = _now()
    count = 0
    pubs = []
    for q in exam.get("questions", []):
        if q.get("has_pending_changes"):
            q["published_at"] = now
            q["version"] = int(q.get("version", 0)) + 1
            q["has_pending_changes"] = False
            q["pending_fields"] = []
            count += 1
            pubs.append(q.get("id") or q.get("number"))
    exam["updated_at"] = now
    _write_exams(exams)
    return {"ok": True, "republished": count, "qids": pubs}


# ─── 题目资源（文件上传）─────────────────────────────────────

@academy_router.get("/api/academy/questions/{qid}/resources")
async def list_resources(qid: str):
    from academy_db import AsyncSessionLocal, QuestionResource
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(QuestionResource).where(QuestionResource.question_id == qid)
        )).scalars().all()
    return [
        {
            "id": r.id, "question_id": r.question_id, "exam_id": r.exam_id,
            "resource_type": r.resource_type, "original_name": r.original_name,
            "url": r.public_url, "file_size": r.file_size, "mime_type": r.mime_type,
        }
        for r in rows
    ]


@academy_router.post("/api/academy/questions/{qid}/resources", status_code=201)
async def upload_resource(
    qid: str,
    file: UploadFile = File(...),
    exam_id: str = Form(""),
):
    from academy_db import AsyncSessionLocal, QuestionResource
    from academy_storage import upload_file

    data = await file.read()
    result = await upload_file(data, file.filename or "upload", prefix=f"academy/q{qid}")

    async with AsyncSessionLocal() as session:
        r = QuestionResource(
            id=uuid.uuid4().hex,
            question_id=qid,
            exam_id=exam_id,
            resource_type=result["resource_type"],
            original_name=file.filename or "",
            s3_key=result["key"],
            local_path=result.get("local_path") or "",
            public_url=result["url"],
            file_size=result["file_size"],
            mime_type=result["mime_type"],
        )
        session.add(r)
        await session.commit()
        rid = r.id

    return {"id": rid, "url": result["url"], "resource_type": result["resource_type"]}


@academy_router.delete("/api/academy/questions/{qid}/resources/{rid}")
async def delete_resource(qid: str, rid: str):
    from academy_db import AsyncSessionLocal, QuestionResource
    from academy_storage import delete_file
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(QuestionResource).where(
                QuestionResource.id == rid,
                QuestionResource.question_id == qid,
            )
        )).scalar_one_or_none()
        if not row:
            raise HTTPException(404, "资源不存在")
        key = row.s3_key
        await session.delete(row)
        await session.commit()

    await delete_file(key)
    return {"ok": True}


# ─── 题目可视化（GeoGebra / Desmos）─────────────────────────

@academy_router.get("/api/academy/questions/{qid}/visualizations")
async def list_visualizations(qid: str):
    from academy_db import AsyncSessionLocal, QuestionVisualization
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(QuestionVisualization).where(QuestionVisualization.question_id == qid)
        )).scalars().all()
    return [
        {
            "id": v.id, "question_id": v.question_id, "viz_type": v.viz_type,
            "title": v.title, "code": v.code, "snapshots": v.snapshots,
            "style_config": v.style_config, "is_primary": v.is_primary,
            "category": v.category,
        }
        for v in rows
    ]


@academy_router.post("/api/academy/questions/{qid}/visualizations", status_code=201)
async def create_visualization(qid: str, body: dict):
    from academy_db import AsyncSessionLocal, QuestionVisualization
    from academy_agents import _parse_snapshots

    vid = uuid.uuid4().hex
    code = body.get("code", "")
    snapshots = _parse_snapshots(code) if code else []

    async with AsyncSessionLocal() as session:
        viz = QuestionVisualization(
            id=vid,
            question_id=qid,
            exam_id=body.get("exam_id", ""),
            viz_type=body.get("viz_type", "geogebra"),
            title=body.get("title", ""),
            code=code,
            snapshots=snapshots,
            style_config=body.get("style_config", {}),
            is_primary=body.get("is_primary", False),
            category=body.get("category", "other"),
        )
        session.add(viz)
        await session.commit()
    return {"id": vid, "snapshots": snapshots}


@academy_router.post("/api/academy/questions/{qid}/visualizations/generate")
async def generate_visualization(qid: str, body: dict):
    """AI 自动为题目生成 GeoGebra 代码"""
    from academy_agents import get_llm, mcp_generate_geogebra_prompt, _parse_snapshots
    from langchain_core.messages import HumanMessage

    category = body.get("category", "function")
    content = body.get("content", "")
    solution = body.get("solution", "")

    prompt_text = mcp_generate_geogebra_prompt(category=category, content=content, solution=solution)
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
    try:
        llm = get_llm()
        resp = await llm.ainvoke([HumanMessage(content=full_prompt)])
        code = resp.content.strip()
    except Exception as e:
        raise HTTPException(502, f"AI 生成失败: {e}")

    snapshots = _parse_snapshots(code)
    return {"code": code, "snapshots": snapshots}


@academy_router.put("/api/academy/questions/{qid}/visualizations/{vid}")
async def update_visualization(qid: str, vid: str, body: dict):
    from academy_db import AsyncSessionLocal, QuestionVisualization
    from academy_agents import _parse_snapshots
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        v = (await session.execute(
            select(QuestionVisualization).where(QuestionVisualization.id == vid)
        )).scalar_one_or_none()
        if not v:
            raise HTTPException(404, "可视化不存在")
        for field in ("title", "code", "style_config", "is_primary", "category", "viz_type"):
            if field in body:
                setattr(v, field, body[field])
        if "code" in body:
            v.snapshots = _parse_snapshots(body["code"])
        await session.commit()
    return {"ok": True}


@academy_router.delete("/api/academy/questions/{qid}/visualizations/{vid}")
async def delete_visualization(qid: str, vid: str):
    from academy_db import AsyncSessionLocal, QuestionVisualization
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        v = (await session.execute(
            select(QuestionVisualization).where(QuestionVisualization.id == vid)
        )).scalar_one_or_none()
        if not v:
            raise HTTPException(404, "可视化不存在")
        await session.delete(v)
        await session.commit()
    return {"ok": True}


# ─── 学生 AI 对话（RAG + SSE 流式）──────────────────────────

@academy_router.post("/api/academy/student/chat")
async def start_or_get_chat(body: dict):
    """创建新会话或获取已有会话 ID"""
    from academy_db import AsyncSessionLocal, ChatSession
    from sqlalchemy import select

    student_id = body.get("student_id", "")
    exam_id    = body.get("exam_id", "")
    session_id = body.get("session_id")

    async with AsyncSessionLocal() as session:
        if session_id:
            row = (await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )).scalar_one_or_none()
            if row:
                return {"session_id": row.id}

        row = ChatSession(
            id=uuid.uuid4().hex,
            student_id=student_id,
            exam_id=exam_id,
            title=body.get("title", "AI 问答"),
        )
        session.add(row)
        await session.commit()
        return {"session_id": row.id}


@academy_router.get("/api/academy/student/chat/stream")
async def chat_stream(
    session_id: str = Query(...),
    question_id: str = Query(""),
    message: str = Query(...),
    exam_id: str = Query(""),
):
    """SSE 流式对话，集成 RAG 检索 + 多轮记忆"""

    async def generate():
        from academy_db import AsyncSessionLocal, ChatMessage
        from academy_agents import rag_search, get_llm
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from sqlalchemy import select

        # RAG 检索
        rag_sources: list[dict] = []
        context_text = ""
        if exam_id:
            try:
                rag_sources = await rag_search(message, exam_id, top_k=5)
                if rag_sources:
                    context_text = "\n\n".join(
                        f"[{s['type']}] {s['content'][:500]}" for s in rag_sources
                    )
            except Exception:
                pass

        system_prompt = (
            "你是智绘书院的 AI 数学助手，专门帮助学生理解数学题目。"
            "回答要清晰、简洁，并适时使用 LaTeX 公式（用 $...$ 或 $$...$$ 包裹）。"
            "请结合下方的对话历史，理解学生当前的疑问上下文。"
        )
        if context_text:
            system_prompt += f"\n\n参考资料（来自本卷）：\n{context_text}"

        # 加载历史消息作为记忆（最多最近 12 条，按时间正序）
        history_msgs = []
        try:
            async with AsyncSessionLocal() as db:
                rows = (await db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(12)
                )).scalars().all()
                rows = list(reversed(rows))
                for r in rows:
                    if r.role == "user":
                        history_msgs.append(HumanMessage(content=r.content or ""))
                    elif r.role == "assistant":
                        history_msgs.append(AIMessage(content=r.content or ""))
        except Exception:
            pass

        messages = [SystemMessage(content=system_prompt)] + history_msgs + [
            HumanMessage(content=message),
        ]

        full_response = ""
        had_error = False
        try:
            llm = get_llm(streaming=True)
            async for chunk in llm.astream(messages):
                delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        except Exception as e:
            had_error = True
            err_msg = f"{type(e).__name__}: {e}"
            yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"

        # 保存消息记录（即使出错也记录用户消息以保留历史）
        try:
            async with AsyncSessionLocal() as db:
                db.add(ChatMessage(
                    id=uuid.uuid4().hex, session_id=session_id,
                    question_id=question_id, role="user", content=message,
                ))
                if full_response:
                    db.add(ChatMessage(
                        id=uuid.uuid4().hex, session_id=session_id,
                        question_id=question_id, role="assistant", content=full_response,
                        rag_sources=rag_sources,
                    ))
                await db.commit()
        except Exception:
            pass

        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── 学生 AI 对话历史 ────────────────────────────────────────

@academy_router.get("/api/academy/student/chat/sessions")
async def list_chat_sessions(
    student_id: str = Query(""),
    exam_id: str = Query(""),
    limit: int = Query(30),
):
    """列出某学生（在某试卷下）的所有对话会话，按最新优先"""
    from academy_db import AsyncSessionLocal, ChatSession, ChatMessage
    from sqlalchemy import select, func

    async with AsyncSessionLocal() as db:
        stmt = select(ChatSession)
        if student_id:
            stmt = stmt.where(ChatSession.student_id == student_id)
        if exam_id:
            stmt = stmt.where(ChatSession.exam_id == exam_id)
        stmt = stmt.order_by(ChatSession.created_at.desc()).limit(limit)
        sessions = (await db.execute(stmt)).scalars().all()

        result = []
        for s in sessions:
            # 取每个会话最后一条消息作为预览
            last = (await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == s.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(1)
            )).scalar_one_or_none()
            count = (await db.execute(
                select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s.id)
            )).scalar() or 0
            result.append({
                "id": s.id,
                "title": s.title or "AI 问答",
                "exam_id": s.exam_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "message_count": count,
                "last_preview": (last.content[:60] if last and last.content else ""),
                "last_role": last.role if last else None,
            })
        return result


@academy_router.get("/api/academy/student/chat/messages")
async def get_chat_messages(session_id: str = Query(...)):
    """获取某会话的全部消息"""
    from academy_db import AsyncSessionLocal, ChatMessage
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )).scalars().all()
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "question_id": r.question_id,
                "rag_sources": r.rag_sources or [],
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


@academy_router.delete("/api/academy/student/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """删除一个会话及其所有消息"""
    from academy_db import AsyncSessionLocal, ChatSession, ChatMessage
    from sqlalchemy import select, delete as sa_delete

    async with AsyncSessionLocal() as db:
        await db.execute(sa_delete(ChatMessage).where(ChatMessage.session_id == session_id))
        await db.execute(sa_delete(ChatSession).where(ChatSession.id == session_id))
        await db.commit()
    return {"ok": True}


# ─── 匿名反馈 ─────────────────────────────────────────────────

@academy_router.post("/api/academy/student/feedback", status_code=201)
async def submit_feedback(body: dict):
    from academy_db import AsyncSessionLocal, StudentFeedback

    async with AsyncSessionLocal() as session:
        fb = StudentFeedback(
            id=uuid.uuid4().hex,
            exam_id=body.get("exam_id", ""),
            question_id=body.get("question_id"),
            student_id=body.get("student_id"),
            anon_token=body.get("anon_token", uuid.uuid4().hex),
            feedback_type=body.get("feedback_type", "suggestion"),
            content=body.get("content", ""),
        )
        session.add(fb)
        await session.commit()
        fid = fb.id
    return {"id": fid, "ok": True}


@academy_router.get("/api/academy/teacher/feedback/{exam_id}")
async def get_feedback(exam_id: str, resolved: Optional[bool] = Query(None)):
    from academy_db import AsyncSessionLocal, StudentFeedback
    from sqlalchemy import select

    stmt = select(StudentFeedback).where(StudentFeedback.exam_id == exam_id)
    if resolved is not None:
        stmt = stmt.where(StudentFeedback.resolved == resolved)
    stmt = stmt.order_by(StudentFeedback.id.desc())

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "id": f.id, "question_id": f.question_id,
            "feedback_type": f.feedback_type, "content": f.content,
            "resolved": f.resolved, "resolved_note": f.resolved_note,
        }
        for f in rows
    ]


@academy_router.put("/api/academy/teacher/feedback/{fid}/resolve")
async def resolve_feedback(fid: str, body: dict):
    from academy_db import AsyncSessionLocal, StudentFeedback
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        fb = (await session.execute(
            select(StudentFeedback).where(StudentFeedback.id == fid)
        )).scalar_one_or_none()
        if not fb:
            raise HTTPException(404, "反馈不存在")
        fb.resolved = True
        fb.resolved_note = body.get("note", "")
        await session.commit()
    return {"ok": True}


# ─── 五阶段 Agent 流水线（SSE）────────────────────────────────

@academy_router.post("/api/academy/exams/{eid}/run-agents")
async def run_agents_pipeline(eid: str):
    """
    触发 LangGraph 五阶段 Agent 流水线，结果以 SSE 实时推送给客户端。
    最终将分析结果回写到 academy_exams.json。
    """

    async def generate():
        exams = _read_exams()
        exam = next((e for e in exams if e["id"] == eid), None)
        if not exam:
            yield f"data: {json.dumps({'error': '试卷不存在'}, ensure_ascii=False)}\n\n"
            return

        # 将题目序列化为 raw_text（LangGraph 的输入）
        raw_text = f"试卷：{exam.get('title','')}\n\n"
        for q in exam.get("questions", []):
            raw_text += f"第{q.get('number','?')}题：{q.get('content','')}\n"
            opts = q.get("options") or []
            for o in opts:
                raw_text += f"  {o}\n"
            raw_text += "\n"

        state = {
            "exam_id": eid,
            "raw_text": raw_text,
            "file_type": "text",
            "questions": exam.get("questions", []),  # 传入已有题目避免重解析
            "errors": [],
            "stage": "init",
            "progress": [],
        }

        try:
            from academy_agents import get_exam_graph
            graph = get_exam_graph()

            # 逐步驱动 LangGraph 图，每个节点完成后推送进度
            async for event in graph.astream(state):
                # event 是 {node_name: updated_state}
                for node_name, updated in event.items():
                    progress = updated.get("progress", [])
                    last_msg = progress[-1] if progress else node_name
                    yield f"data: {json.dumps({'stage': node_name, 'message': last_msg}, ensure_ascii=False)}\n\n"
                    # 更新 state 供下一节点
                    state = updated

        except ImportError:
            # LangGraph 未安装：逐节点手动执行
            from academy_agents import (
                parse_node, classify_node, solve_node, visualize_node, validate_node
            )
            for node_fn in [parse_node, classify_node, solve_node, visualize_node, validate_node]:
                state = await node_fn(state)
                progress = state.get("progress", [])
                last_msg = progress[-1] if progress else node_fn.__name__
                yield f"data: {json.dumps({'stage': node_fn.__name__, 'message': last_msg}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return

        # 将 Agent 结果回写到 JSON
        final_qs = state.get("questions", [])
        if final_qs:
            exam["questions"] = final_qs
            exam["agent_analyzed"] = True
            exam["updated_at"] = _now()
            _write_exams(exams)

        yield f"data: {json.dumps({'done': True, 'question_count': len(final_qs)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── RAG 索引 ────────────────────────────────────────────────

@academy_router.post("/api/academy/exams/{eid}/index-rag")
async def index_exam_rag(eid: str):
    exams = _read_exams()
    exam = next((e for e in exams if e["id"] == eid), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    try:
        from academy_agents import index_exam_for_rag
        count = await index_exam_for_rag(exam)
        return {"ok": True, "indexed_chunks": count}
    except Exception as e:
        raise HTTPException(502, f"RAG 索引失败: {e}")


# ─── 微调数据集导出 ────────────────────────────────────────────

@academy_router.get("/api/academy/finetune/export")
async def export_finetune():
    """将 FineTuneRecord + 高质量 QuestionVisualization 导出为 JSONL"""
    from academy_db import AsyncSessionLocal, FineTuneRecord, QuestionVisualization
    from sqlalchemy import select

    lines = []
    async with AsyncSessionLocal() as session:
        # 已标记的 FineTuneRecord
        for row in (await session.execute(select(FineTuneRecord))).scalars().all():
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": row.question_content},
                    {"role": "assistant", "content": row.viz_code},
                ],
                "category": row.category,
                "viz_type": row.viz_type,
                "quality_score": row.quality_score,
            }, ensure_ascii=False))

        # in_finetune_dataset 标记的 QuestionVisualization
        for v in (await session.execute(
            select(QuestionVisualization).where(QuestionVisualization.in_finetune_dataset == True)
        )).scalars().all():
            lines.append(json.dumps({
                "messages": [
                    {"role": "user",      "content": f"[category:{v.category}] 生成GeoGebra可视化"},
                    {"role": "assistant", "content": v.code},
                ],
                "category": v.category,
                "viz_type": v.viz_type,
            }, ensure_ascii=False))

    content = "\n".join(lines)
    return StreamingResponse(
        iter([content]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=finetune_dataset.jsonl"},
    )


# ─── 管理员：LangSmith 追踪指标代理 ──────────────────────────

@academy_router.get("/api/academy/admin/langsmith-stats")
async def langsmith_stats(limit: int = Query(20, le=100)):
    """代理请求 LangSmith API，返回最近运行的统计信息。"""
    import httpx

    ls_api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    ls_project = os.environ.get("LANGCHAIN_PROJECT", "default")
    ls_url     = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    tracing    = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if not ls_api_key or not tracing:
        return {
            "enabled": False,
            "message": "LangSmith 未启用（需设置 LANGCHAIN_TRACING_V2=true 及 LANGCHAIN_API_KEY）",
            "project": ls_project,
        }

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            # 获取项目列表，找到对应 project_id
            proj_r = await client.get(
                f"{ls_url}/api/v1/projects",
                headers={"x-api-key": ls_api_key},
                params={"name": ls_project},
            )
            projects = proj_r.json() if proj_r.is_success else []
            project_id = projects[0]["id"] if isinstance(projects, list) and projects else None

            runs_data: list[dict] = []
            if project_id:
                runs_r = await client.get(
                    f"{ls_url}/api/v1/runs",
                    headers={"x-api-key": ls_api_key},
                    params={
                        "session": project_id,
                        "run_type": "chain",
                        "limit": limit,
                        "select": "id,name,status,start_time,end_time,total_tokens,error",
                    },
                )
                if runs_r.is_success:
                    runs_data = runs_r.json() or []

        # 统计
        total   = len(runs_data)
        success = sum(1 for r in runs_data if r.get("status") == "success")
        errors  = total - success
        total_tokens = sum(r.get("total_tokens") or 0 for r in runs_data)

        latencies: list[float] = []
        for r in runs_data:
            s, e = r.get("start_time"), r.get("end_time")
            if s and e:
                try:
                    dt = (
                        datetime.fromisoformat(e.replace("Z", "+00:00")) -
                        datetime.fromisoformat(s.replace("Z", "+00:00"))
                    ).total_seconds()
                    latencies.append(dt)
                except Exception:
                    pass
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        return {
            "enabled": True,
            "project": ls_project,
            "project_id": project_id,
            "dashboard_url": f"https://smith.langchain.com/o/public/projects/p/{project_id}" if project_id else f"https://smith.langchain.com",
            "total_runs": total,
            "success_runs": success,
            "error_runs": errors,
            "success_rate": round(success / total * 100, 1) if total else 0,
            "total_tokens": total_tokens,
            "avg_latency_sec": avg_latency,
            "recent_runs": [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "status": r.get("status"),
                    "start_time": r.get("start_time"),
                    "total_tokens": r.get("total_tokens"),
                    "error": r.get("error"),
                }
                for r in runs_data[:10]
            ],
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


# ─── 管理员：全局反馈概览 ────────────────────────────────────

@academy_router.get("/api/academy/admin/feedback-overview")
async def admin_feedback_overview(limit: int = Query(100, le=500)):
    """跨所有试卷的反馈概览，管理员专用。"""
    from academy_db import AsyncSessionLocal, StudentFeedback
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        stmt = select(StudentFeedback).order_by(StudentFeedback.id.desc()).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()

    exams = _read_exams()
    exam_title = {e["id"]: e.get("title", e["id"]) for e in exams}
    # 题目内容 + 题号映射
    qcontent: dict[str, str] = {}
    qnumber: dict[str, int | str] = {}
    for e in exams:
        for idx, q in enumerate(e.get("questions", [])):
            qid = str(q.get("id", ""))
            if qid:
                qcontent[qid] = str(q.get("content", ""))[:100]
                qnumber[qid] = q.get("number") or (idx + 1)

    type_count: dict[str, int] = {}
    for f in rows:
        type_count[f.feedback_type] = type_count.get(f.feedback_type, 0) + 1

    return {
        "total": len(rows),
        "type_count": type_count,
        "unresolved": sum(1 for f in rows if not f.resolved),
        "items": [
            {
                "id": f.id,
                "exam_id": f.exam_id,
                "exam_title": exam_title.get(f.exam_id, f.exam_id or "未知"),
                "question_id": f.question_id,
                "question_number": qnumber.get(f.question_id or ""),
                "question_preview": qcontent.get(f.question_id or "", ""),
                "feedback_type": f.feedback_type,
                "content": f.content,
                "resolved": f.resolved,
            }
            for f in rows
        ],
    }
