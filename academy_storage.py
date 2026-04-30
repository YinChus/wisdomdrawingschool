"""
智绘书院 — 文件存储模块
优先使用腾讯云 COS（S3 兼容），未配置时回退到本地目录。
所需环境变量（来自 .env.local）：
  S3_ENDPOINT_URL  S3_BUCKET_NAME  S3_REGION
  S3_ACCESS_KEY    S3_SECRET_KEY
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import uuid
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────

S3_ENDPOINT  = os.getenv("S3_ENDPOINT_URL", "")
S3_BUCKET    = os.getenv("S3_BUCKET_NAME", "")
S3_REGION    = os.getenv("S3_REGION", "ap-shanghai")
S3_ACCESS    = os.getenv("S3_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", ""))
S3_SECRET    = os.getenv("S3_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", ""))

LOCAL_DIR = Path(os.getenv("LOCAL_UPLOAD_DIR", "F:/dataset-annotator/data/academy_files"))
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

USE_S3 = bool(S3_ENDPOINT and S3_BUCKET and S3_ACCESS and S3_SECRET)

if USE_S3:
    import boto3
    from botocore.config import Config as BotoConfig

    _s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS,
        aws_secret_access_key=S3_SECRET,
        region_name=S3_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )
else:
    _s3 = None  # type: ignore[assignment]


# ─── 工具函数 ─────────────────────────────────────────────────

def _gen_key(prefix: str, original_name: str) -> str:
    ext = Path(original_name).suffix.lower() or ""
    return f"{prefix}/{uuid.uuid4().hex}{ext}"


def _detect_type(original_name: str) -> str:
    """根据文件名检测资源类型"""
    ext = Path(original_name).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}:
        return "image"
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}:
        return "video"
    if ext in {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}:
        return "audio"
    if ext in {".glb", ".gltf", ".obj", ".fbx", ".stl", ".dae"}:
        return "model_3d"
    if ext in {".pdf"}:
        return "pdf"
    return "other"


# ─── 核心接口 ─────────────────────────────────────────────────

async def upload_file(
    data: bytes,
    original_name: str,
    prefix: str = "academy",
) -> dict:
    """
    上传文件，返回:
      { key, url, local_path, resource_type, mime_type, file_size }
    """
    key = _gen_key(prefix, original_name)
    mime = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    rtype = _detect_type(original_name)
    size = len(data)

    if USE_S3:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _s3.put_object(  # type: ignore[union-attr]
                Bucket=S3_BUCKET,
                Key=key,
                Body=data,
                ContentType=mime,
                ACL="public-read",
            ),
        )
        url = f"{S3_ENDPOINT.rstrip('/')}/{S3_BUCKET}/{key}"
        local_path = None
    else:
        # 本地存储：将 key 中的 / 替换为 _ 避免目录问题
        safe_name = key.replace("/", "__")
        local_path_obj = LOCAL_DIR / safe_name
        local_path_obj.write_bytes(data)
        local_path = str(local_path_obj)
        url = f"/api/academy/files/{safe_name}"

    return {
        "key": key,
        "url": url,
        "local_path": local_path,
        "resource_type": rtype,
        "mime_type": mime,
        "file_size": size,
    }


async def get_file_url(key: str) -> str:
    if USE_S3:
        return f"{S3_ENDPOINT.rstrip('/')}/{S3_BUCKET}/{key}"
    safe_name = key.replace("/", "__")
    return f"/api/academy/files/{safe_name}"


async def delete_file(key: str) -> None:
    if USE_S3:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _s3.delete_object(Bucket=S3_BUCKET, Key=key),  # type: ignore[union-attr]
        )
    else:
        safe_name = key.replace("/", "__")
        p = LOCAL_DIR / safe_name
        if p.exists():
            p.unlink()


async def read_file_bytes(key: str) -> bytes:
    if USE_S3:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: _s3.get_object(Bucket=S3_BUCKET, Key=key),  # type: ignore[union-attr]
        )
        return resp["Body"].read()
    else:
        safe_name = key.replace("/", "__")
        return (LOCAL_DIR / safe_name).read_bytes()


def get_local_path(key: str) -> Path:
    """返回本地文件路径（仅非 S3 模式有效）"""
    safe_name = key.replace("/", "__")
    return LOCAL_DIR / safe_name
