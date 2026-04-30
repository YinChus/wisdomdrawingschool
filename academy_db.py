"""
智绘书院 — 异步数据库模型（SQLAlchemy 2.0 async）
默认使用 SQLite（开箱即用），设置 DATABASE_URL 环境变量切换到 PostgreSQL：
  postgresql+asyncpg://user:pass@host:5432/academy
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON, Index
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool, NullPool

# ─── 数据库连接 ───────────────────────────────────────────────

_DB_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///F:/dataset-annotator/data/academy.db"
)
_IS_SQLITE = "sqlite" in _DB_URL

_engine_kwargs: dict = {"echo": False}
if _IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["poolclass"] = StaticPool
else:
    _engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(_DB_URL, **_engine_kwargs)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def gen_id() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─── 模型定义 ─────────────────────────────────────────────────

class QuestionResource(Base):
    """每道题上传的附件：视频 / 音频 / 图片 / 3D 模型 等"""
    __tablename__ = "question_resources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    question_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    exam_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(20))   # image|video|audio|model_3d|model_2d|pdf|other
    original_name: Mapped[str] = mapped_column(String(500))
    s3_key: Mapped[Optional[str]] = mapped_column(String(500))
    local_path: Mapped[Optional[str]] = mapped_column(String(500))
    public_url: Mapped[Optional[str]] = mapped_column(String(1000))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


class QuestionVisualization(Base):
    """GeoGebra / Desmos 可视化（含分步快照，用于步骤回放）"""
    __tablename__ = "question_visualizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    question_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    exam_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    viz_type: Mapped[str] = mapped_column(String(20), default="geogebra")   # geogebra|desmos
    title: Mapped[Optional[str]] = mapped_column(String(200))
    code: Mapped[str] = mapped_column(Text, default="")          # GeoGebra 命令或 Desmos state JSON
    snapshots: Mapped[Optional[list]] = mapped_column(JSON)       # [{step, description, commands, full_xml?}]
    style_config: Mapped[Optional[dict]] = mapped_column(JSON)    # {color, bg, ggb_color}
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    category: Mapped[Optional[str]] = mapped_column(String(30))   # math category for style
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    in_finetune_dataset: Mapped[bool] = mapped_column(Boolean, default=False)


class StudentFeedback(Base):
    """学生匿名反馈（不会的题目 / 有错误 / 建议）"""
    __tablename__ = "student_feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    exam_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    question_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    student_id: Mapped[Optional[str]] = mapped_column(String(36))   # 可为空（匿名）
    anon_token: Mapped[Optional[str]] = mapped_column(String(64))   # 匿名追踪 token
    feedback_type: Mapped[str] = mapped_column(String(50))          # dont_understand|too_hard|has_error|suggestion
    content: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_note: Mapped[Optional[str]] = mapped_column(Text)


class ChatSession(Base):
    """学生 AI 问答会话"""
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    student_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    exam_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


class ChatMessage(Base):
    """问答消息记录"""
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    question_id: Mapped[Optional[str]] = mapped_column(String(36))   # 关联到哪道题
    role: Mapped[str] = mapped_column(String(20))                     # user|assistant
    content: Mapped[str] = mapped_column(Text)
    rag_sources: Mapped[Optional[list]] = mapped_column(JSON)         # 引用的 RAG 片段
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


class RagChunk(Base):
    """RAG 向量索引（用题目 + 解析生成，供学生问答检索）"""
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    exam_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    question_id: Mapped[Optional[str]] = mapped_column(String(36))
    chunk_type: Mapped[str] = mapped_column(String(50))   # question|solution|answer|exam_title
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Optional[list]] = mapped_column(JSON)   # float[] (cosine similarity)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


class FineTuneRecord(Base):
    """可视化代码写入微调数据集"""
    __tablename__ = "finetune_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_id)
    question_id: Mapped[str] = mapped_column(String(36))
    question_content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))
    viz_type: Mapped[str] = mapped_column(String(20))
    viz_code: Mapped[str] = mapped_column(Text)
    style_config: Mapped[Optional[dict]] = mapped_column(JSON)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)


# ─── 工具函数 ─────────────────────────────────────────────────

async def create_tables() -> None:
    """应用启动时创建所有表"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session
