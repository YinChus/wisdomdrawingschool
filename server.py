"""
数学可视化数据集标注系统 - 后端服务
存储路径: F:/dataset-annotator/data/
运行: pip install fastapi uvicorn python-multipart && python server.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 存储目录 ──────────────────────────────────────────────────
BASE_DIR   = Path("F:/dataset-annotator/data")
IMAGES_DIR = BASE_DIR / "images"
EXPORT_DIR = BASE_DIR / "exports"
DB_FILE    = BASE_DIR / "triplets.json"

for d in [BASE_DIR, IMAGES_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 类型定义 ──────────────────────────────────────────────────

CATEGORIES = [
    "solid_geometry", "function", "conic_curve",
    "trigonometric", "probability", "algebra",
    "vector", "sequence", "other",
]

class StyleFeatures(BaseModel):
    has_axes: bool = False
    label_count: int = 0
    fill_present: bool = False
    shape_complexity: str = "medium"   # low / medium / high
    line_count: int = 0

class DatasetImage(BaseModel):
    id: str
    name: str
    filename: str   # 本地文件名
    dim: str        # "2d" | "3d"
    view_angle: str = ""

class DatasetCode(BaseModel):
    id: str
    engine: str     # "geogebra" | "desmos3d" | "threejs"
    dim: str        # "2d" | "3d"
    code: str
    description: str = ""
    image_refs: List[str] = []   # image id 列表
    is_normalized: bool = False

class DatasetTriplet(BaseModel):
    id: str
    seq: int
    category: str = "function"
    dim: str = "2d"          # 题目整体维度：2d | 3d
    question_text: str = ""
    question_latex: str = ""
    solution_text: str = ""
    solution_latex: str = ""
    images: List[DatasetImage] = []
    codes: List[DatasetCode] = []
    style_features: StyleFeatures = StyleFeatures()
    annotation_status: str = "draft"   # draft / annotated / reviewed
    annotator: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

# ── 数据库读写 ─────────────────────────────────────────────────

def read_db() -> List[dict]:
    if not DB_FILE.exists():
        return []
    text = DB_FILE.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return json.loads(text)

def write_db(data: List[dict]) -> None:
    # 原子写入：先写临时文件，再重命名，避免并发读到空文件
    tmp = DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, DB_FILE)

def next_seq(data: List[dict]) -> int:
    return max((d["seq"] for d in data), default=0) + 1

# ── 风格特征自动识别 ───────────────────────────────────────────

def auto_detect_style_features(code: str, engine: str) -> dict:
    """
    从可视化代码中自动提取风格特征。
    支持 GeoGebra 脚本（逐行命令）和 Desmos 3D JSON state。
    """
    import re as _re

    features = {
        "has_axes": False,
        "fill_present": False,
        "label_count": 0,
        "line_count": 0,
        "shape_complexity": "low",
    }

    if engine in ("desmos3d", "desmos2d"):
        # ── Desmos JSON state 解析 ──
        try:
            state = json.loads(code)
            exprs = state.get("expressions", {}).get("list", [])
            label_n = 0
            line_n  = 0
            fill_n  = 0
            for e in exprs:
                t = e.get("type", "")
                latex = e.get("latex", "")
                if e.get("showLabel"):
                    label_n += 1
                if e.get("fillOpacity") or e.get("fill"):
                    fill_n += 1
                if t in ("expression",) and any(k in latex for k in ["\\line", "\\segment", "\\vec"]):
                    line_n += 1
                if "x=" in latex or "y=" in latex or "z=" in latex:
                    features["has_axes"] = True
            features["label_count"] = label_n
            features["line_count"]  = line_n
            features["fill_present"] = fill_n > 0
            total = len(exprs)
            features["shape_complexity"] = "high" if total > 12 else ("medium" if total > 5 else "low")
        except Exception:
            pass  # 非 JSON 则跳过
    else:
        # ── GeoGebra 逐行命令解析 ──
        lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("//")]
        total = len(lines)

        axis_kw   = ["SetAxesVisible", "ShowAxes", "xAxis", "yAxis", "zAxis",
                     "Axes(", "SetAxis", "GridType("]
        fill_kw   = ["SetFilling(", "SetBackgroundColor(", "SetColor(",
                     "Polygon(", "CircleSector(", "Arc("]
        label_kw  = ["SetLabel(", "SetLabelMode(", "Text[", "LaTeX[",
                     "SetCaption(", ":=", "SetValue("]
        line_kw   = ["Line(", "Line[", "Segment(", "Segment[",
                     "Ray(", "Ray[", "Vector(", "Vector[",
                     "InfiniteLine(", "HalfLine(", "PolyLine("]

        def count_kw(kws):
            return sum(1 for l in lines if any(k in l for k in kws))

        has_axes   = any(any(k in l for k in axis_kw) for l in lines)
        fill_count = count_kw(fill_kw)
        label_count = count_kw(label_kw)
        line_count  = count_kw(line_kw)

        features["has_axes"]     = has_axes
        features["fill_present"] = fill_count > 0
        features["label_count"]  = label_count
        features["line_count"]   = line_count
        features["shape_complexity"] = "high" if total > 20 else ("medium" if total > 8 else "low")

    return features


def merge_style_features(existing: dict, detected: dict) -> dict:
    """检测结果与现有人工标注合并，人工标注优先（若非默认值）。"""
    merged = dict(detected)
    # 如果人工标注了非默认值，保留人工值
    if existing.get("label_count", 0) > 0:
        merged["label_count"] = existing["label_count"]
    if existing.get("line_count", 0) > 0:
        merged["line_count"] = existing["line_count"]
    if existing.get("shape_complexity", "low") != "low":
        merged["shape_complexity"] = existing["shape_complexity"]
    if existing.get("has_axes"):
        merged["has_axes"] = True
    if existing.get("fill_present"):
        merged["fill_present"] = True
    return merged

# ── FastAPI ───────────────────────────────────────────────────

app = FastAPI(title="数学可视化数据集标注系统")


@app.on_event("startup")
async def _startup():
    """初始化 SQLAlchemy 数据库表（新功能模块）"""
    try:
        from academy_db import create_tables
        await create_tables()
    except Exception as e:
        print(f"[Academy DB] 初始化失败（非致命）: {e}")


try:
    from academy_api import academy_router
    app.include_router(academy_router)
except Exception as _e:
    print(f"[Academy API] 路由加载失败（非致命）: {_e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（存放上传的图片）
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# 静态资源（本地JS/CSS库）
_STATIC_DIR = Path("F:/dataset-annotator/static")
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── 三元组接口 ─────────────────────────────────────────────────

@app.post("/api/render-tikz")
async def render_tikz(body: dict):
    """
    代理 QuickLaTeX API 渲染 TikZ 代码，返回图片 URL。
    QuickLaTeX 是免费服务，无需 API key，支持完整 TikZ。
    """
    import urllib.request, urllib.parse
    code = body.get("code", "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="code is required")

    # 如果没有 \begin{tikzpicture}，自动包裹
    if "\\begin{tikzpicture}" not in code:
        code = "\\begin{tikzpicture}\n" + code + "\n\\end{tikzpicture}"

    # 最小 preamble：只加 tikz 核心库，避免引入不存在的包报错
    # 根据代码内容动态追加可选包
    preamble_lines = ["\\usepackage{tikz}", "\\usepackage{amsmath,amssymb}"]
    if "pgfplots" in code or "\\begin{axis}" in code:
        preamble_lines.append("\\usepackage{pgfplots}")
        preamble_lines.append("\\pgfplotsset{compat=1.18}")
    if "tdplot" in code or "tikz-3dplot" in code:
        preamble_lines.append("\\usepackage{tikz-3dplot}")
    # common tikz libraries
    libs = []
    if "arrows" in code:       libs.append("arrows.meta")
    if "calc" in code:         libs.append("calc")
    if "patterns" in code:     libs.append("patterns")
    if "decorations" in code:  libs.append("decorations.pathmorphing")
    if "matrix" in code:       libs.append("matrix")
    if "positioning" in code:  libs.append("positioning")
    if "shapes" in code:       libs.append("shapes.geometric")
    if "fit" in code:          libs.append("fit")
    if libs:
        preamble_lines.append("\\usetikzlibrary{" + ",".join(libs) + "}")
    preamble = "\n".join(preamble_lines)

    # IMPORTANT: use quote_via=urllib.parse.quote (not quote_plus) so spaces
    # in TikZ option strings are encoded as %20, not + which breaks pgfkeys.
    params = urllib.parse.urlencode(
        {
            "formula":  code,
            "fsize":    "20px",
            "fcolor":   "000000",
            "mode":     "0",
            "out":      "1",
            "remhost":  "quicklatex.com",
            "preamble": preamble,
            "rnd":      str(id(code)),
        },
        quote_via=urllib.parse.quote,
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://quicklatex.com/latex3.f",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            result = resp.read().decode("utf-8").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"QuickLaTeX 请求失败: {e}")

    # QuickLaTeX 响应格式：
    # 成功：第1行 "0"，第2行 "<url> <x> <w> <h>"
    # 失败：第1行 "1"，后续行为错误信息
    lines = result.splitlines()
    status_code_ql = lines[0].strip() if lines else "1"
    if status_code_ql == "0" and len(lines) >= 2:
        url = lines[1].split()[0]
        return {"ok": True, "url": url}
    else:
        err = "\n".join(lines[1:]) if len(lines) > 1 else result
        return {"ok": False, "error": err}


@app.get("/api/triplets")
def list_triplets(status: Optional[str] = None, category: Optional[str] = None):
    data = read_db()
    if status:
        data = [d for d in data if d["annotation_status"] == status]
    if category:
        data = [d for d in data if d["category"] == category]

    raw = read_db()
    stats = {
        "total": len(raw),
        "draft": sum(1 for d in raw if d["annotation_status"] == "draft"),
        "annotated": sum(1 for d in raw if d["annotation_status"] == "annotated"),
        "reviewed": sum(1 for d in raw if d["annotation_status"] == "reviewed"),
        "by_category": {cat: sum(1 for d in raw if d["category"] == cat) for cat in CATEGORIES},
    }
    return {"triplets": data, "stats": stats}

@app.post("/api/triplets", status_code=201)
def create_triplet(body: dict):
    data = read_db()
    seq  = next_seq(data)
    tid  = f"q{seq:03d}"
    now  = datetime.now().isoformat()

    triplet: dict = {
        "id": tid, "seq": seq,
        "category": body.get("category", "function"),
        "question_text": body.get("question_text", ""),
        "question_latex": body.get("question_latex", ""),
        "images": [],
        "codes": [],
        "style_features": StyleFeatures().model_dump(),
        "annotation_status": "draft",
        "annotator": body.get("annotator", ""),
        "notes": "",
        "created_at": now,
        "updated_at": now,
    }
    data.append(triplet)
    write_db(data)
    return {"triplet": triplet}

@app.get("/api/triplets/{tid}")
def get_triplet(tid: str):
    data = read_db()
    item = next((d for d in data if d["id"] == tid), None)
    if not item:
        raise HTTPException(404, "not found")
    return {"triplet": item}

@app.put("/api/triplets/{tid}")
def update_triplet(tid: str, body: dict):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404, "not found")

    protected = {"id", "seq", "created_at"}
    for k, v in body.items():
        if k not in protected:
            data[idx][k] = v
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"triplet": data[idx]}

@app.delete("/api/triplets/{tid}")
def delete_triplet(tid: str):
    data = read_db()
    new_data = [d for d in data if d["id"] != tid]
    if len(new_data) == len(data):
        raise HTTPException(404, "not found")
    # 删除关联图片文件
    old = next(d for d in data if d["id"] == tid)
    for img in old.get("images", []):
        p = IMAGES_DIR / img["filename"]
        if p.exists():
            p.unlink()
    write_db(new_data)
    return {"ok": True}

# ── 图片上传接口 ───────────────────────────────────────────────

@app.post("/api/triplets/{tid}/images")
async def upload_image(
    tid: str,
    file: UploadFile = File(...),
    dim: str = Form("2d"),
    view_angle: str = Form(""),
    custom_name: str = Form(""),
):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404, "triplet not found")

    ext      = Path(file.filename or "img.png").suffix.lower() or ".png"
    img_id   = str(uuid.uuid4())[:8]
    # 命名规则: {tid}_{img_id}{ext}  如 q001_a3f2b1.png
    filename = f"{tid}_{img_id}{ext}"
    dest     = IMAGES_DIR / filename

    content = await file.read()
    dest.write_bytes(content)

    img_record = {
        "id":         img_id,
        "name":       custom_name or filename,
        "filename":   filename,
        "dim":        dim,
        "view_angle": view_angle,
    }
    data[idx]["images"].append(img_record)
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"image": img_record, "url": f"/images/{filename}"}

@app.delete("/api/triplets/{tid}/images/{img_id}")
def delete_image(tid: str, img_id: str):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404, "triplet not found")

    images = data[idx]["images"]
    img    = next((im for im in images if im["id"] == img_id), None)
    if not img:
        raise HTTPException(404, "image not found")

    p = IMAGES_DIR / img["filename"]
    if p.exists():
        p.unlink()

    data[idx]["images"] = [im for im in images if im["id"] != img_id]
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"ok": True}

@app.patch("/api/triplets/{tid}/images/{img_id}")
def patch_image(tid: str, img_id: str, body: dict):
    """修改图片元数据（dim / name / view_angle）"""
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404, "triplet not found")
    images = data[idx]["images"]
    img = next((im for im in images if im["id"] == img_id), None)
    if img is None:
        raise HTTPException(404, "image not found")
    allowed = {"dim", "name", "view_angle"}
    for k, v in body.items():
        if k in allowed:
            img[k] = v
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"image": img}

# ── 代码段接口 ─────────────────────────────────────────────────

@app.post("/api/triplets/{tid}/codes")
def add_code(tid: str, body: dict):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404, "triplet not found")

    code_record = {
        "id":           str(uuid.uuid4())[:8],
        "engine":       body.get("engine", "geogebra"),
        "dim":          body.get("dim", "2d"),
        "code":         body.get("code", ""),
        "description":  body.get("description", ""),
        "image_refs":   body.get("image_refs", []),
        "is_normalized": body.get("is_normalized", False),
    }
    data[idx]["codes"].append(code_record)
    # 自动识别风格特征并合并到 triplet
    detected = auto_detect_style_features(code_record["code"], code_record["engine"])
    existing = data[idx].get("style_features", {})
    data[idx]["style_features"] = merge_style_features(existing, detected)
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"code": code_record}

@app.put("/api/triplets/{tid}/codes/{code_id}")
def update_code(tid: str, code_id: str, body: dict):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404)
    codes = data[idx]["codes"]
    cidx  = next((i for i, c in enumerate(codes) if c["id"] == code_id), -1)
    if cidx == -1:
        raise HTTPException(404)
    for k, v in body.items():
        if k != "id":
            codes[cidx][k] = v
    # 重新自动识别风格特征
    detected = auto_detect_style_features(codes[cidx]["code"], codes[cidx]["engine"])
    existing = data[idx].get("style_features", {})
    data[idx]["style_features"] = merge_style_features(existing, detected)
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"code": codes[cidx]}

@app.delete("/api/triplets/{tid}/codes/{code_id}")
def delete_code(tid: str, code_id: str):
    data = read_db()
    idx  = next((i for i, d in enumerate(data) if d["id"] == tid), -1)
    if idx == -1:
        raise HTTPException(404)
    data[idx]["codes"] = [c for c in data[idx]["codes"] if c["id"] != code_id]
    data[idx]["updated_at"] = datetime.now().isoformat()
    write_db(data)
    return {"ok": True}

# ── 导出接口 ───────────────────────────────────────────────────

@app.post("/api/export")
def export_dataset(body: dict):
    """
    按比例拆分为 train / val / test 并导出为 JSONL 对话格式
    body: { train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1 }
    """
    import random

    data = read_db()
    # 只导出 reviewed 的条目（可配置）
    reviewed = [d for d in data if d["annotation_status"] in ("reviewed", "annotated")]
    if not reviewed:
        raise HTTPException(400, "没有已标注数据可导出")

    random.shuffle(reviewed)
    n     = len(reviewed)
    tr    = body.get("train_ratio", 0.8)
    vr    = body.get("val_ratio", 0.1)
    n_tr  = max(1, int(n * tr))
    n_val = max(0, int(n * vr))

    splits = {
        "train": reviewed[:n_tr],
        "val":   reviewed[n_tr:n_tr + n_val],
        "test":  reviewed[n_tr + n_val:],
    }

    def to_conversation(triplet: dict) -> List[dict]:
        """每个 (question, code) 对 → 一条对话样本"""
        samples = []
        sf = triplet.get("style_features", {})
        style_hint = (
            f"坐标轴:{'有' if sf.get('has_axes') else '无'} "
            f"标注数:{sf.get('label_count', 0)} "
            f"填充:{'有' if sf.get('fill_present') else '无'} "
            f"复杂度:{sf.get('shape_complexity', 'medium')} "
            f"线段数:{sf.get('line_count', 0)}"
        )
        for code in triplet.get("codes", []):
            if not code.get("code", "").strip():
                continue
            samples.append({
                "id": f"{triplet['id']}_{code['id']}",
                "category": triplet["category"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是数学可视化专家，按题型生成符合规范色板的可视化代码。\n"
                            "色板规范：主要几何对象蓝色(#3b82f6)、顶点/焦点琥珀(#f59e0b)、"
                            "辅助线灰色(#9ca3af)、对比曲线红色(#ef4444)、向量紫色(#8b5cf6)。\n"
                            f"引擎：{code['engine']}，维度：{code['dim']}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"【题型】{triplet['category']}\n"
                            f"【题目】{triplet.get('question_latex') or triplet.get('question_text', '')}\n"
                            f"【解析】{triplet.get('solution_latex') or triplet.get('solution_text', '') or '（未填写）'}\n"
                            f"【风格要求】{style_hint}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": code["code"],
                    },
                ],
            })
        return samples

    counts = {}
    for split_name, items in splits.items():
        out_path = EXPORT_DIR / f"{split_name}.jsonl"
        lines = []
        for item in items:
            lines.extend(to_conversation(item))
        out_path.write_text(
            "\n".join(json.dumps(l, ensure_ascii=False) for l in lines),
            encoding="utf-8",
        )
        counts[split_name] = len(lines)

    return {
        "ok": True,
        "export_dir": str(EXPORT_DIR),
        "counts": counts,
    }

@app.get("/api/export/download/{split}")
def download_export(split: str):
    if split not in ("train", "val", "test"):
        raise HTTPException(400, "invalid split")
    p = EXPORT_DIR / f"{split}.jsonl"
    if not p.exists():
        raise HTTPException(404, "请先执行导出")
    return FileResponse(str(p), filename=f"{split}.jsonl", media_type="application/octet-stream")

# ── 前端页面 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>请将 ui.html 放在同目录下</h1>")

@app.get("/geometry-board.html", response_class=HTMLResponse)
def serve_geo_board():
    html_path = Path(__file__).parent / "geometry-board.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>geometry-board.html 未找到</h1>", status_code=404)

@app.get("/ai-settings.html", response_class=HTMLResponse)
def serve_ai_settings():
    html_path = Path(__file__).parent / "ai-settings.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ai-settings.html 未找到</h1>", status_code=404)

@app.get("/academy.html", response_class=HTMLResponse)
def serve_academy():
    html_path = Path(__file__).parent / "academy.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy.html 未找到</h1>", status_code=404)

@app.get("/academy-login.html", response_class=HTMLResponse)
def serve_academy_login():
    html_path = Path(__file__).parent / "academy-login.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy-login.html 未找到</h1>", status_code=404)

@app.get("/academy-student.html", response_class=HTMLResponse)
def serve_academy_student():
    html_path = Path(__file__).parent / "academy-student.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy-student.html 未找到</h1>", status_code=404)

@app.get("/academy-admin.html", response_class=HTMLResponse)
def serve_academy_admin():
    html_path = Path(__file__).parent / "academy-admin.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy-admin.html 未找到</h1>", status_code=404)

# ══════════════════════════════════════════════════════════════
#  智绘书院 — 试卷解析 & 教学资源生成模块
# ══════════════════════════════════════════════════════════════

import base64
import re as _re

ACADEMY_DB_FILE   = BASE_DIR / "academy_exams.json"
ACADEMY_CFG_FILE  = BASE_DIR / "academy_config.json"
ACADEMY_FILES_DIR = BASE_DIR / "academy_files"
ACADEMY_FILES_DIR.mkdir(parents=True, exist_ok=True)

# ── 分类规则（移植自 TypeScript classifier.ts）────────────────

_MATH_RULES = [
    (r"函数|f\s*\(|y\s*=|定义域|值域|单调", "function"),
    (r"不等式|解不等", "inequality"),
    (r"复数|实部|虚部", "complex_number"),
    (r"三角形|正弦定理|余弦定理", "triangle_solution"),
    (r"数列|等差|等比|通项", "sequence"),
    (r"概率|期望|方差", "probability"),
    (r"棱柱|棱锥|球|体积|表面积|二面角|空间", "solid_geometry"),
    (r"椭圆|双曲线|抛物线", "conic_curve"),
    (r"导数|求导|极值|最大值|最小值.*f", "derivative"),
    (r"sin\b|cos\b|tan\b|正弦|余弦|三角", "trigonometric"),
    (r"向量", "vector"),
    (r"集合|交集|并集", "set_logic"),
]

_VIZ_MAP = {
    "function":          {"viz_type": "function_graph", "engine": "geogebra", "desc": "函数图像（GeoGebra）"},
    "inequality":        {"viz_type": "function_graph", "engine": "geogebra", "desc": "不等式图像（GeoGebra）"},
    "solid_geometry":    {"viz_type": "geometry_3d",    "engine": "threejs",  "desc": "3D立体几何（Three.js）"},
    "conic_curve":       {"viz_type": "conic_curve",    "engine": "geogebra", "desc": "圆锥曲线（GeoGebra）"},
    "trigonometric":     {"viz_type": "trig_graph",     "engine": "geogebra", "desc": "三角函数图像（GeoGebra）"},
    "probability":       {"viz_type": "data_chart",     "engine": "desmos3d", "desc": "概率统计图表"},
    "vector":            {"viz_type": "vector_diagram", "engine": "geogebra", "desc": "向量图示（GeoGebra）"},
    "derivative":        {"viz_type": "function_graph", "engine": "geogebra", "desc": "导函数图像（GeoGebra）"},
    "sequence":          {"viz_type": "data_chart",     "engine": "geogebra", "desc": "数列图示（GeoGebra）"},
    "triangle_solution": {"viz_type": "geometry_2d",    "engine": "geogebra", "desc": "三角形图示（GeoGebra）"},
    "complex_number":    {"viz_type": "complex_plane",  "engine": "geogebra", "desc": "复数平面（GeoGebra）"},
    "set_logic":         {"viz_type": "venn_diagram",   "engine": "geogebra", "desc": "韦恩图（GeoGebra）"},
}

# ── 配置读写 ──────────────────────────────────────────────────

def read_academy_cfg() -> dict:
    if ACADEMY_CFG_FILE.exists():
        try:
            return json.loads(ACADEMY_CFG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "text_model": os.environ.get("LLM_TEXT_MODEL", os.environ.get("LLM_MODEL", "")),
        "vision_model": os.environ.get("LLM_VISION_MODEL", os.environ.get("LLM_MODEL", "")),
        "geo_model": os.environ.get("LLM_GEO_MODEL", ""),
    }

def write_academy_cfg(cfg: dict) -> None:
    ACADEMY_CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

# ── 试卷数据库 ─────────────────────────────────────────────────

def read_academy_db() -> List[dict]:
    if not ACADEMY_DB_FILE.exists():
        return []
    text = ACADEMY_DB_FILE.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []

def write_academy_db(data: List[dict]) -> None:
    tmp = ACADEMY_DB_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, ACADEMY_DB_FILE)

# ── LLM 调用 ──────────────────────────────────────────────────

async def _call_llm(
    messages: list,
    *,
    model: str = "",
    max_tokens: int = 4096,
    response_json: bool = False,
    temperature: float = 0.2,
) -> str:
    """调用 OpenAI 兼容 API，返回第一条 assistant 消息文本。"""
    cfg = read_academy_cfg()
    api_key  = cfg.get("api_key", "")
    base_url = (cfg.get("base_url", "") or "").rstrip("/")
    if not api_key or not base_url:
        raise HTTPException(503, "请先在「智绘书院」设置中配置 LLM_API_KEY 和 LLM_BASE_URL")
    if not model:
        model = cfg.get("text_model", "") or ""
    if not model:
        raise HTTPException(503, "请先在「智绘书院」设置中配置文本模型名称")

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"LLM 调用失败: {e}") from e

# ── 题目分类（纯规则，不依赖LLM）────────────────────────────

def _classify_question(content: str) -> str:
    for pattern, category in _MATH_RULES:
        if _re.search(pattern, content):
            return category
    return "other"

# ── 文件内容提取 ───────────────────────────────────────────────

# ── 答案文本智能切分到每道题 ─────────────────────────────────

async def _distribute_answers_to_questions(exam: dict, full_answer_text: str) -> dict:
    """
    将整卷答案文本（含 "第N题" / "N. xxx" / "1)" 等）切分并写入每道题的
    answer / solution 字段。优先用正则；正则切不到题号时回退到 LLM。
    返回 {distributed, skipped}。会原地修改 exam["questions"]。
    """
    import re as _re
    questions = exam.get("questions", [])
    if not questions or not full_answer_text:
        return {"distributed": 0, "skipped": len(questions)}

    text = full_answer_text.replace("\r\n", "\n").strip()

    # 1) 正则切分：匹配 "第N题"/"N."/"N、"/"(N)"/"N)" 作为段落起点
    pattern = _re.compile(
        r"(?:^|\n)\s*(?:第\s*)?(\d{1,3})\s*(?:题|[\.．、)）])\s*[：: ]?",
        flags=_re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    parsed: dict[int, str] = {}
    if matches:
        for i, m in enumerate(matches):
            try:
                num = int(m.group(1))
            except Exception:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = text[start:end].strip()
            if seg:
                parsed[num] = seg

    # 2) 若正则切不到 ≥ 一半题目，调用 LLM 做更稳健的切分
    need_llm = len(parsed) < max(1, len(questions) // 2)
    if need_llm:
        try:
            sys_prompt = (
                "你是数学老师助手。下面是一份试卷的整卷答案/解析文本，"
                "请把它按题号切分，输出严格的 JSON：{\"answers\":[{\"number\":1,\"answer\":\"...\",\"solution\":\"...\"},...]}。"
                "answer 仅放最终答案（如 A、x=2、(1,3) 等），solution 放详细推导/解析步骤。"
                "必须完整保留原文中的每步推导、公式、LaTeX 记号，不要总结或省略。"
                "若原文某题只有最终答案而无推导过程，solution 留空字符串。"
            )
            user_prompt = f"题目数量约 {len(questions)} 题。整卷答案文本如下：\n\n{text[:20000]}"
            raw = await _call_llm(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user_prompt}],
                max_tokens=3072, response_json=True,
            )
            obj = json.loads(raw) if isinstance(raw, str) else raw
            llm_parsed: dict[int, dict] = {}
            for it in (obj.get("answers") or []):
                try:
                    n = int(it.get("number"))
                    llm_parsed[n] = {
                        "answer": str(it.get("answer", "")).strip(),
                        "solution": str(it.get("solution", "")).strip(),
                    }
                except Exception:
                    continue
            if llm_parsed:
                # LLM 结果优先；正则结果只在 LLM 未覆盖到时用作 solution 兜底
                distributed = 0
                for idx, q in enumerate(questions):
                    qn = q.get("number") or (idx + 1)
                    try:
                        qn_int = int(qn)
                    except Exception:
                        continue
                    if qn_int in llm_parsed:
                        ans = llm_parsed[qn_int]["answer"]
                        sol = llm_parsed[qn_int]["solution"] or parsed.get(qn_int, "")
                        if ans and not q.get("answer"):
                            q["answer"] = ans
                        if sol and not q.get("solution"):
                            q["solution"] = sol
                        if ans or sol:
                            distributed += 1
                return {"distributed": distributed, "skipped": len(questions) - distributed}
        except Exception:
            pass  # 回退到正则结果

    # 3) 用正则切分结果填充：把整段当作 solution；首行短答案当 answer
    distributed = 0
    for idx, q in enumerate(questions):
        qn = q.get("number") or (idx + 1)
        try:
            qn_int = int(qn)
        except Exception:
            continue
        seg = parsed.get(qn_int)
        if not seg:
            continue
        # 首行 / 30 字以内当 answer
        first_line = seg.split("\n", 1)[0].strip()
        ans_guess = first_line if len(first_line) <= 30 else ""
        if ans_guess and not q.get("answer"):
            q["answer"] = ans_guess
        # 只要有内容就写入 solution（即使与 answer 重复，至少让学生有东西看）
        if not q.get("solution") and seg:
            q["solution"] = seg
        distributed += 1

    return {"distributed": distributed, "skipped": len(questions) - distributed}


async def _extract_file_content(file_bytes: bytes, filename: str, cfg: dict) -> dict:
    """
    根据文件类型提取内容，返回 {mode: 'text'|'image_b64', content: str, vision_model: str}
    """
    ext = Path(filename).suffix.lower()

    # ── Word ──
    if ext in (".docx", ".doc"):
        try:
            import docx  # python-docx
            import io
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if text.strip():
                return {"mode": "text", "content": text}
        except ImportError:
            pass
        except Exception:
            pass
        # 回退：直接告诉LLM是Word文档
        encoded = base64.b64encode(file_bytes).decode()
        return {"mode": "text", "content": f"[Word文档内容，共{len(file_bytes)}字节，请根据上下文解析题目]"}

    # ── PDF ──
    if ext == ".pdf":
        try:
            import pymupdf  # PyMuPDF
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()
            text = "\n".join(pages_text).strip()
            if text and len(text) > 100:
                # 不再做粗暴截断，由 _llm_parse_exam 分块并行处理
                return {"mode": "text", "content": text[:60000]}
        except ImportError:
            pass
        except Exception:
            pass
        # 回退：用 PDF 第一页渲染为图片（需 pymupdf）或直接用 base64
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            page = doc[0]
            pix = page.get_pixmap(dpi=120)
            img_bytes = pix.tobytes("png")
            doc.close()
            encoded = base64.b64encode(img_bytes).decode()
            return {
                "mode": "image_b64",
                "content": f"data:image/png;base64,{encoded}",
                "vision_model": cfg.get("vision_model", cfg.get("text_model", "")),
            }
        except Exception:
            pass
        # 最后回退：base64 PDF（部分视觉API可接受）
        encoded = base64.b64encode(file_bytes).decode()
        return {
            "mode": "image_b64",
            "content": f"data:application/pdf;base64,{encoded}",
            "vision_model": cfg.get("vision_model", cfg.get("text_model", "")),
        }

    # ── 图片 ──
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/png")
    encoded = base64.b64encode(file_bytes).decode()
    return {
        "mode": "image_b64",
        "content": f"data:{mime};base64,{encoded}",
        "vision_model": cfg.get("vision_model", cfg.get("text_model", "")),
    }

# ── 解析试卷 ──────────────────────────────────────────────────

_PARSE_SYSTEM_PROMPT = """你是专业的高中数学试卷解析专家。
请从试卷内容中提取所有题目，严格返回 JSON 格式，不要包含任何其他文字。

返回格式：
{
  "title": "试卷标题（如无则为空字符串）",
  "subject": "数学",
  "questions": [
    {
      "number": 1,
      "type": "choice",
      "content": "完整题目文本，数学公式用 $...$ 或 $$...$$ 包裹",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "答案（如有，否则为空）",
      "score": 5
    }
  ]
}

type 字段说明：
- choice：单选题
- multi_choice：多选题
- fill：填空题
- calculation：解答/计算/证明题
"""

async def _llm_parse_exam(extracted: dict, cfg: dict, answer_text: str = "") -> dict:
    """调用 LLM 解析试卷，返回结构化 JSON。answer_text 为可选的答案解析文件文本内容。

    优化：当文本模式且内容较长时，按"题号"启发式切分并发并行 LLM 调用，最后合并题目，
    显著降低单次大请求的等待时间。
    """
    import asyncio

    answer_hint = ""  # 不再在解析阶段注入答案文本（避免占用 max_tokens 导致 JSON 截断，完整题目丢失）。
    # 答案/解析会在解析后由 _distribute_answers_to_questions 独立处理。
    _ = answer_text  # 保留参数以兼容调用处

    # ── 图片模式：视觉模型一次性解析（无法切分）──
    if extracted["mode"] != "text":
        vision_model = extracted.get("vision_model") or cfg.get("vision_model") or cfg.get("text_model", "")
        user_text = "请识别并解析图片中的试卷题目。" + answer_hint
        messages = [
            {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": extracted["content"]}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        raw = await _call_llm(messages, model=vision_model, max_tokens=8192, response_json=True)
        return _safe_parse_json(raw)

    # ── 文本模式：分块并行 ──
    text = extracted["content"]
    chunks = _split_exam_by_questions(text, target_chars=2500, max_chunks=8)

    # 单块直接走原流程
    if len(chunks) <= 1:
        messages = [
            {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请解析以下试卷内容：\n\n{text}{answer_hint}"},
        ]
        model = cfg.get("text_model", "")
        raw = await _call_llm(messages, model=model, max_tokens=8192, response_json=True)
        return _safe_parse_json(raw)

    # 多块：并行调用
    model = cfg.get("text_model", "")

    async def _parse_one(idx: int, chunk: str) -> list:
        msgs = [
            {"role": "system", "content": _PARSE_SYSTEM_PROMPT},
            {"role": "user", "content":
                f"以下是试卷的第 {idx+1}/{len(chunks)} 段（按题号切分），请只解析其中包含的题目，"
                f"保持每题的原始题号。{answer_hint}\n\n{chunk}"},
        ]
        try:
            raw = await _call_llm(msgs, model=model, max_tokens=8192, response_json=True)
            obj = _safe_parse_json(raw)
            return obj.get("questions", []) or []
        except Exception:
            return []

    results = await asyncio.gather(*[_parse_one(i, c) for i, c in enumerate(chunks)])

    # 合并：按题号去重
    merged: list = []
    seen_numbers = set()
    for qlist in results:
        for q in qlist:
            num = q.get("number")
            if num and num in seen_numbers:
                continue
            if num:
                seen_numbers.add(num)
            merged.append(q)

    # 重新按题号排序（缺失题号则保持原顺序）
    merged.sort(key=lambda q: (q.get("number") or 9999))

    return {"title": "", "subject": "数学", "questions": merged}


def _safe_parse_json(raw: str) -> dict:
    """从 LLM 原始输出中安全提取 JSON。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        raise HTTPException(422, f"LLM 返回格式非 JSON：{raw[:300]}")


def _split_exam_by_questions(text: str, target_chars: int = 2500, max_chunks: int = 8) -> list[str]:
    """按行匹配题号（如 '1.'、'1、'、'(1)'、'第1题' 等）将试卷文本切成多段。

    每段尽量接近 target_chars，超过后下一道题归入新段；最多切 max_chunks 段。
    若无法识别题号则按字符长度均分。
    """
    if len(text) <= target_chars:
        return [text]

    # 题号起始行的正则
    qpat = _re.compile(r"^\s*(?:第\s*\d+\s*题|[\(（]?\s*\d+\s*[\)）.、]|\d+\s*[\.。、])")

    # 收集每个题号起始位置
    lines = text.split("\n")
    starts = [0]  # 字符偏移
    offset = 0
    for i, line in enumerate(lines):
        if i > 0 and qpat.match(line):
            starts.append(offset)
        offset += len(line) + 1  # +1 for \n

    if len(starts) < 2:
        # 无题号，按长度均分
        n = max(2, min(max_chunks, len(text) // target_chars + 1))
        size = len(text) // n + 1
        return [text[i:i+size] for i in range(0, len(text), size)]

    # 把题号起始位置打包成段
    chunks = []
    cur_start = 0
    for s in starts[1:]:
        if s - cur_start >= target_chars:
            chunks.append(text[cur_start:s])
            cur_start = s
    chunks.append(text[cur_start:])

    # 限制最大段数：超过则按顺序合并
    while len(chunks) > max_chunks:
        # 合并最短的相邻两段
        idx = min(range(len(chunks) - 1), key=lambda i: len(chunks[i]) + len(chunks[i+1]))
        chunks[idx] = chunks[idx] + chunks[idx+1]
        del chunks[idx+1]

    return chunks


# ── API: 配置 ─────────────────────────────────────────────────

@app.get("/api/academy/settings")
def get_academy_settings():
    cfg = read_academy_cfg()
    # 隐藏 key 中间部分
    key = cfg.get("api_key", "")
    if key and len(key) > 8:
        display_key = key[:4] + "****" + key[-4:]
    else:
        display_key = "（未配置）" if not key else key
    return {
        "api_key_preview": display_key,
        "base_url": cfg.get("base_url", ""),
        "text_model": cfg.get("text_model", ""),
        "vision_model": cfg.get("vision_model", ""),
        "geo_model": cfg.get("geo_model", ""),
        "configured": bool(key and cfg.get("base_url")),
    }

@app.put("/api/academy/settings")
def update_academy_settings(body: dict):
    cfg = read_academy_cfg()
    for k in ("api_key", "base_url", "text_model", "vision_model", "geo_model"):
        if k in body and isinstance(body[k], str):
            cfg[k] = body[k].strip()
    write_academy_cfg(cfg)
    return {"ok": True}

@app.post("/api/academy/test-connection")
async def test_academy_connection():
    """测试 LLM 连接是否正常。"""
    try:
        result = await _call_llm(
            [{"role": "user", "content": "回复ok即可"}],
            max_tokens=10,
        )
        cfg = read_academy_cfg()
        return {"ok": True, "response": result, "model": cfg.get("text_model", "")}
    except HTTPException as e:
        return {"ok": False, "error": e.detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/academy/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """通用文件 → 文本抽取（用于解析文件上传等场景）。"""
    try:
        file_bytes = await file.read()
        cfg = read_academy_cfg()
        extracted = await _extract_file_content(file_bytes, file.filename or "file", cfg)
        if extracted.get("mode") == "text":
            return {"text": extracted.get("content", ""), "mode": "text"}
        # 图片模式：返回提示，前端无法直接预览为文本
        return {"text": "[已上传图片/扫描件，请使用 AI 一键解题或手动整理]", "mode": "image"}
    except Exception as e:
        return {"text": "", "error": str(e)}


@app.post("/api/academy/draw-ai")
async def draw_ai_generate(body: dict):
    """
    根据用户描述 + 题目内容，调用 LLM 生成 GeoGebra 2D / 3D 或 TikZ 绘图代码。
    body: {
      board_type: 'geogebra'|'desmos3d'|'tikz',
      prompt: str,
      context?: str,            # 已有代码（追加场景）
      question_content?: str,   # 当前题目正文，用于结合题意生成
      question_answer?: str,
    }
    returns: { ok: bool, commands: str, explanation: str }
    """
    board_type = body.get("board_type", "geogebra")
    prompt = (body.get("prompt", "") or "").strip()
    if not prompt:
        raise HTTPException(400, "请输入绘图描述")

    if board_type == "geogebra":
        system_msg = (
            "你是一个专业的数学可视化助手，精通 GeoGebra 6 / Classic 命令语法。\n"
            "请根据【题目内容】和用户的补充描述，生成贴合该题的 GeoGebra 2D 命令序列。\n\n"
            "⚠️ 必须严格遵守 GeoGebra 真实语法（以下违规命令一定执行失败，禁止生成）：\n"
            "1. **必须显式乘号**：写 `2*p` 不能写 `2p`；写 `3*x` 不能写 `3x`。\n"
            "2. **禁止命名参数 / 双等号方程**：\n"
            "   ❌ `Parabola(Focus=(p/2,0), Directrix=x=p/(-2))` —— GeoGebra 不支持 `Focus=` `Directrix=` 这种关键字参数。\n"
            "   ❌ `C = y^2 = 2*p*x` —— GeoGebra 不允许双等号方程。\n"
            "   ❌ `Line((1,0), Slope=Undefined)` —— 没有 `Slope=Undefined` 这种东西。\n"
            "3. **抛物线 y²=2px（顶点原点、开口向右）正确写法**——\n"
            "   ✅ **首选** `c=ImplicitCurve(y^2 - 2*p*x)` —— 它能与 Line 做 `Intersect(c, L, n)` 取第 n 个交点。\n"
            "   ✅ 也可 `c=Parabola((p/2, 0), Line((-p/2, 0), (-p/2, 1)))` —— Parabola 只接受 `(焦点, 准线)` 两个参数，准线必须是 Line 对象。\n"
            "   ⚠️ **不推荐** `c=Curve(t^2/(2*p), t, t, -8, 8)` —— 参数曲线 Curve 与 Line 的 `Intersect(c, L, n)` 行为异常，n 参数对参数曲线不可靠。\n"
            "   规则：**只要后面要做 Intersect 取第 n 个交点，抛物线/椭圆/双曲线一律用 ImplicitCurve 写**。\n"
            "4. **过点 P 垂直于 x 轴的直线**：用 `L=Line(P, yAxis)`，不要用 `Slope=Undefined` 或 `(P.x+1, P.y)` 这种伪表达式。\n"
            "5. **垂线**：`PerpendicularLine(<点>, <直线>)`，不是 `Perpendicular(...)`。\n"
            "6. **重心**：`Centroid(<多边形>)`，参数必须是已经定义的 Polygon，不能是三个点。\n"
            "   先 `t=Polygon(O,A,B)`，再 `G=Centroid(t)`。\n"
            "7. **不要用 `P.x` `P.y`**：要取点 P 的横纵坐标用 `x(P)` `y(P)`。\n"
            "8. **Curve 必须 5 个参数**：`Curve(<x表达式>, <y表达式>, <参数名>, <下界>, <上界>)`。\n"
            "9. **Intersect**：两个参数都必须是几何对象（线/曲线/圆）；返回第 n 个交点用 `Intersect(<对象1>, <对象2>, n)`。\n"
            "10. **变量名**只用字母数字下划线，不要中文/空格；点用大写 `A,B,C`，曲线/直线用小写 `c,line1,L`。\n"
            "11. **不要使用 Markdown 代码块标记**（``` 之类），直接输出纯文本命令。\n"
            "12. **不要在命令行末尾写注释**：注释必须独占一行，以 `#` 开头；\n"
            "    ❌ `L1=Line(P,(1/2,0))  # 水平直线`  ✅ `# 水平直线\\nL1=Line(P,(1/2,0))`\n"
            "13. **数学符号必须用 ASCII**：写 `sqrt(p/2)` 不要写 `√(p/2)`；写 `pi` 不要写 `π`；写 `*` 不要写 `×`。\n"
            "14. **变量名不能复用**：`Curve(t^2/(2*p), t, t, -6, 6)` 已经把 `t` 注册为曲线参数，\n"
            "    后面就不能再 `t=Polygon(...)`；请改用 `tri=Polygon(...)` 之类不冲突的名字。\n\n"
            "输出格式：\n"
            "1. 每行一条命令，用 `# 步骤N: 描述` 作为分节注释\n"
            "2. 先输出命令块，再用 `---说明---` 分隔，最后用中文解释每一步与题目的关系\n"
            "3. 命令块只输出纯命令和注释，不要加 markdown 代码块标记\n\n"
            "正确示例（抛物线 y²=2px 与垂直于 x 轴的直线交于 A、B）：\n"
            "# 步骤1: 定义参数与抛物线（用 ImplicitCurve 才能配合 Intersect 取第 n 个交点）\n"
            "p=1\n"
            "c=ImplicitCurve(y^2 - 2*p*x)\n"
            "# 步骤2: 过 (1,0) 作垂直于 x 轴的直线\n"
            "L=Line((1,0), yAxis)\n"
            "# 步骤3: 求交点\n"
            "A=Intersect(c, L, 1)\n"
            "B=Intersect(c, L, 2)\n"
            "# 步骤4: 三角形 OAB 与重心\n"
            "O=(0,0)\n"
            "tri=Polygon(O,A,B)\n"
            "G=Centroid(tri)\n"
            "---说明---\n"
            "..."
        )
    elif board_type == "tikz":
        system_msg = (
            "你是一个专业的数学排版助手，精通 LaTeX TikZ 绘图。\n"
            "请根据【题目内容】和用户的补充描述，生成贴合该题的 TikZ 代码。\n"
            "输出格式要求：\n"
            "1. 输出完整 \\begin{tikzpicture}...\\end{tikzpicture} 块，可直接编译\n"
            "2. 使用清晰的坐标系、合理的 scale；常用 \\draw, \\filldraw, \\node, \\path\n"
            "3. 必要时加 # 注释行说明每步含义（TikZ 使用 % 注释）\n"
            "4. 先输出代码块，再用 ---说明--- 分隔，最后用中文解释每步与题目的关系\n"
            "5. 代码块只输出纯 TikZ，不要加 markdown 代码块标记\n"
            "示例：\n\\begin{tikzpicture}[scale=1]\n  \\draw[->] (-3,0)--(3,0) node[right]{$x$};\n  \\draw[->] (0,-3)--(0,3) node[above]{$y$};\n  \\draw (0,0) circle (2);\n\\end{tikzpicture}\n---说明---\n绘制了..."
        )
    else:  # geogebra 3d
        system_msg = (
            "你是一个专业的3D数学可视化助手，精通 GeoGebra 3D 计算器的命令语法。\n"
            "请根据【题目内容】和用户的补充描述，生成贴合该题的 GeoGebra 3D 命令序列。\n"
            "输出格式要求：\n"
            "1. 每行一条命令，用 # 步骤N: 描述 作为分节注释\n"
            "2. 使用 GeoGebra 3D 命令，如：Sphere((0,0,0),1)、Cone((0,0,0),(0,0,3),2)、\n"
            "   f(x,y)=sin(x)*cos(y)、Plane(A,B,C)、IntersectPath(f,g) 等\n"
            "3. 坐标默认三维格式 (x,y,z)\n"
            "4. 先输出命令块，再用 ---说明--- 分隔，最后用中文解释每步与题目的关系\n"
            "5. 命令块只输出纯命令和注释，不要加 markdown 代码块标记\n"
            "示例：\n"
            "# 步骤1: 单位球\nSphere((0,0,0),1)\n# 步骤2: 抛物面\nf(x,y)=x^2+y^2\n---说明---\n步骤1创建了以原点为心、半径为1的球..."
        )

    context = body.get("context", "") or ""
    qc = (body.get("question_content", "") or "").strip()
    qa = (body.get("question_answer", "") or "").strip()

    parts_user = []
    if qc:
        parts_user.append(f"【题目内容】\n{qc}")
    if qa:
        parts_user.append(f"【参考答案】\n{qa}")
    if context:
        parts_user.append(f"【已有代码（请在此基础上修改/扩展）】\n{context}")
    parts_user.append(f"【绘图需求】\n{prompt}")
    user_content = "\n\n".join(parts_user)


    try:
        raw = await _call_llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1500,
            temperature=0.0,   # 画图代码必须确定性，避免 LLM 编造伪 API
        )
    except HTTPException as e:
        return {"ok": False, "error": e.detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # 分离命令块和说明
    parts = raw.split("---说明---")
    commands = parts[0].strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""

    return {"ok": True, "commands": commands, "explanation": explanation}


# ── 严谨模式（4 阶段多 Agent）────────────────────────────────
# 阶段 1 Planner: 题目→结构化绘图计划 JSON
# 阶段 2 Coder:   计划→GeoGebra 命令
# 阶段 3 Executor: 由前端 evalCommand 完成（无法在后端跑）
# 阶段 4 Healer:  失败行 + 已成功对象 → LLM 局部修补
# 参考: DiagrammerGPT (COLM 2024, arXiv:2310.12128) planner-auditor feedback loop

_GEOGEBRA_RULES_BRIEF = (
    "GeoGebra 真实语法红线（违反必失败）：\n"
    "- 必须显式 `*`（写 `2*p` 不是 `2p`）\n"
    "- 禁止命名参数（无 `Focus=` `Slope=Undefined`）和双等号（无 `c=y^2=2*p*x`）\n"
    "- 抛物线/椭圆/双曲线后续若要 Intersect 取第 n 交点，必须用 `ImplicitCurve(...)` \n"
    "  例：`c=ImplicitCurve(y^2 - 2*p*x)`\n"
    "- 过点垂直 x 轴：`Line(P, yAxis)`\n"
    "- 垂线：`PerpendicularLine(<点>, <直线>)`\n"
    "- 重心必须 `Centroid(<Polygon对象>)`，先 `tri=Polygon(O,A,B)` 再 `G=Centroid(tri)`\n"
    "- 取坐标用 `x(P)` `y(P)` 不是 `P.x`\n"
    "- `Curve` 必须 5 参数：`Curve(x表达式, y表达式, 参变量, 下界, 上界)`\n"
    "- `Intersect(<对象1>, <对象2>, n)` 两参必须是几何对象\n"
    "- 变量名只用字母数字下划线；同名不可复用（用过 `t` 当 Curve 参变量后不能再 `t=Polygon(...)`，改 `tri`）\n"
    "- 禁用 markdown ```、行末注释、Unicode（用 `sqrt` `pi` `*` 不用 `√` `π` `×`）\n"
    "- 注释必须独占一行，以 `#` 开头\n"
)


@app.post("/api/academy/draw-multi/plan-and-code")
async def draw_multi_plan_and_code(body: dict):
    """
    严谨模式 阶段 1+2：Planner（题目→结构化计划）→ Coder（计划→GeoGebra 代码）。
    body: { question_content, question_answer, prompt }
    return: { ok, plan: [...], code: "...", explanation: "..." }
    """
    qc = (body.get("question_content", "") or "").strip()
    qa = (body.get("question_answer", "") or "").strip()
    prompt = (body.get("prompt", "") or "").strip()
    if not qc and not prompt:
        raise HTTPException(400, "请至少提供题目内容或绘图描述")

    # ───── 阶段 1: Planner ─────
    planner_sys = (
        "你是数学几何"
        "题目分析专家。请把【题目】拆解成结构化【绘图计划 JSON】，只描述画什么、不写代码。\n\n"
        "输出严格的 JSON（无 markdown、无解释），格式：\n"
        '{ "scene": "一句话场景描述", \n'
        '  "objects": [\n'
        '    {"id":"标识符", "kind":"对象类型", "purpose":"用途说明", ...其它字段...},\n'
        '    ...\n'
        "  ] }\n\n"
        "支持的 kind 与字段（必须严格使用）：\n"
        '  number       value            如 {"id":"p","kind":"number","value":"2"}\n'
        '  point        expr             如 {"id":"F","kind":"point","expr":"(p/2, 0)"}\n'
        '  conic        expr             一般二次曲线，expr 形如 "y^2 - 2*p*x"（=0 形式，不写=0）\n'
        '  parabola_std open, p          标准抛物线，open ∈ right/left/up/down\n'
        '  ellipse_std  a, b             标准椭圆 x^2/a^2 + y^2/b^2 = 1\n'
        '  hyperbola_std a, b, axis      axis ∈ x/y\n'
        '  circle       center, radius   或 expr (圆方程)\n'
        '  line         p1, p2           或 through+direction (direction:"vertical"|"horizontal"|表达式)\n'
        '  segment      p1, p2\n'
        '  vector       p1, p2\n'
        '  function     expr             如 "sin(x)"\n'
        '  intersection of:[id1,id2], index  取第 n 个交点（n=1 时可省略）\n'
        '  midpoint     of:[id1,id2]\n'
        '  perpendicular point, line     过点垂直于 line\n'
        '  parallel     point, line\n'
        '  polygon      vertices:[id...] \n'
        '  centroid     of:polygonId\n'
        '  angle        of:[id1,id2,id3] \n\n'
        "规则：\n"
        "1. id 严格按依赖顺序排列（被引用对象必须在前）。\n"
        "2. id 只用字母数字下划线，点用大写（A、B、F），其它用小写（c、l1、tri）。\n"
        "3. 抛物线/椭圆/双曲线如果后续要做 intersection，请优先用 `conic` 或 `*_std`，会被翻译成 ImplicitCurve。\n"
        "4. 同名变量不要复用。\n"
        "5. 只列必要对象，不要装饰性元素。\n"
    )
    user_for_planner_parts = []
    if qc: user_for_planner_parts.append(f"【题目】\n{qc}")
    if qa: user_for_planner_parts.append(f"【参考答案】\n{qa}")
    if prompt: user_for_planner_parts.append(f"【补充要求】\n{prompt}")
    user_for_planner = "\n\n".join(user_for_planner_parts)

    try:
        plan_raw = await _call_llm(
            [{"role":"system","content":planner_sys},
             {"role":"user","content":user_for_planner}],
            max_tokens=1200, temperature=0.3, response_json=True,
        )
    except HTTPException as e:
        return {"ok": False, "stage":"planner", "error": e.detail}
    except Exception as e:
        return {"ok": False, "stage":"planner", "error": str(e)}

    try:
        plan = json.loads(plan_raw)
    except Exception:
        # LLM 没返回纯 JSON，尝试抓第一个 {...}
        m = _re.search(r"\{[\s\S]+\}", plan_raw)
        if not m:
            return {"ok": False, "stage":"planner", "error":"Planner 未返回有效 JSON", "raw": plan_raw[:500]}
        try:
            plan = json.loads(m.group(0))
        except Exception as e:
            return {"ok": False, "stage":"planner", "error": f"Planner JSON 解析失败: {e}", "raw": plan_raw[:500]}

    # ───── 阶段 2: Coder ─────
    coder_sys = (
        "你是 GeoGebra 命令翻译器。把给定的【绘图计划 JSON】严格翻译成 GeoGebra 6 命令序列。\n"
        "禁止增删对象、禁止改名、禁止发挥创造。每个 object 翻译成一行命令，按 JSON 顺序输出。\n\n"
        + _GEOGEBRA_RULES_BRIEF +
        "\n翻译模板（必须严格遵守）：\n"
        '  number       → `<id>=<value>`\n'
        '  point        → `<id>=<expr>`\n'
        '  conic        → `<id>=ImplicitCurve(<expr>)`\n'
        '  parabola_std → 根据 open: right→`<id>=ImplicitCurve(y^2 - 2*<p>*x)` / left→`y^2 + 2*<p>*x` / up→`x^2 - 2*<p>*y` / down→`x^2 + 2*<p>*y`\n'
        '  ellipse_std  → `<id>=ImplicitCurve(x^2/(<a>)^2 + y^2/(<b>)^2 - 1)`\n'
        '  hyperbola_std→ axis=x: `<id>=ImplicitCurve(x^2/(<a>)^2 - y^2/(<b>)^2 - 1)`；axis=y: `... y^2/(<a>)^2 - x^2/(<b>)^2 - 1`\n'
        '  circle       → 有 center+radius: `<id>=Circle(<center>, <radius>)`；有 expr: `<id>=ImplicitCurve(<expr>)`\n'
        '  line         → 有 p1+p2: `<id>=Line(<p1>, <p2>)`；through+vertical: `<id>=Line(<through>, yAxis)`；through+horizontal: `<id>=Line(<through>, xAxis)`；through+表达式: `<id>=Line(<through>, <方向向量或斜率>)`\n'
        '  segment      → `<id>=Segment(<p1>, <p2>)`\n'
        '  vector       → `<id>=Vector(<p1>, <p2>)`\n'
        '  function     → `<id>(x)=<expr>`\n'
        '  intersection → `<id>=Intersect(<of[0]>, <of[1]>, <index|1>)`\n'
        '  midpoint     → `<id>=Midpoint(<of[0]>, <of[1]>)`\n'
        '  perpendicular→ `<id>=PerpendicularLine(<point>, <line>)`\n'
        '  parallel     → `<id>=Line(<point>, <line>)`  (GeoGebra 的 Line(点,直线) 即过该点的平行线)\n'
        '  polygon      → `<id>=Polygon(<vertices...逗号分隔>)`\n'
        '  centroid     → `<id>=Centroid(<of>)`\n'
        '  angle        → `<id>=Angle(<of[0]>, <of[1]>, <of[2]>)`\n'
        "\n输出格式：\n"
        "  - 每个对象前一行 `# <id>: <purpose>`\n"
        "  - 然后是命令本身\n"
        "  - 整体最后输出 `---说明---` 再用一段中文说明绘图思路（一两句话）\n"
        "  - 不要 markdown ```\n"
    )
    try:
        coder_raw = await _call_llm(
            [{"role":"system","content":coder_sys},
             {"role":"user","content":"【绘图计划 JSON】\n"+json.dumps(plan, ensure_ascii=False, indent=2)}],
            max_tokens=1500, temperature=0.0,
        )
    except HTTPException as e:
        return {"ok": False, "stage":"coder", "plan": plan, "error": e.detail}
    except Exception as e:
        return {"ok": False, "stage":"coder", "plan": plan, "error": str(e)}

    parts = coder_raw.split("---说明---")
    code = parts[0].strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""
    return {"ok": True, "plan": plan, "code": code, "explanation": explanation}


@app.post("/api/academy/draw-multi/heal")
async def draw_multi_heal(body: dict):
    """
    严谨模式 阶段 4：Healer。给定原代码 + 失败行 + 已成功对象列表，让 LLM 仅修补失败行。
    body: {
      plan?: [...],          # 阶段 1 的计划（可选，提供时质量更好）
      code: str,             # 当前代码（含成功+失败的全部行）
      failed_lines: [str],   # 失败的命令行（前端传来的原始命令）
      success_objects: [str],# 已经创建成功的 GeoGebra 对象名（前端 getAllObjectNames 提供）
      attempt: int,          # 第几次修复，从 1 开始
    }
    return: { ok, code: str, changed_lines: [...], explanation: str }
    """
    code = (body.get("code", "") or "").strip()
    failed_lines: list = body.get("failed_lines") or []
    success_objects: list = body.get("success_objects") or []
    plan = body.get("plan")
    attempt = int(body.get("attempt") or 1)
    if not code or not failed_lines:
        return {"ok": False, "error": "缺少 code 或 failed_lines"}

    healer_sys = (
        "你是 GeoGebra 修复器。下面是一段命令脚本，部分行执行失败。\n"
        "你的任务：**只修改失败行**，输出整段修补后的代码（保留原顺序、注释、成功的行原样不动）。\n\n"
        + _GEOGEBRA_RULES_BRIEF +
        "\n常见失败原因排查清单：\n"
        "1. 引用了未定义对象 → 看『已成功对象』清单，调整名字或顺序\n"
        "2. 命名参数 / 双等号方程 → 改写为 ImplicitCurve\n"
        "3. Curve 与 Line 的 Intersect(.,.,n) 不可靠 → 把 Curve(...) 换成 ImplicitCurve(...) 同等价\n"
        "4. 隐式乘法 → 补 `*`\n"
        "5. P.x / P.y → 改 x(P) / y(P)\n"
        "6. 变量名复用 → 重命名（如 t→tri）\n"
        "7. Centroid 接受了点而非 Polygon → 先建 Polygon\n"
        "\n输出要求：\n"
        "- 直接输出整段修补后的代码（纯文本，无 markdown 包裹）\n"
        "- 保留原有注释行\n"
        "- 末尾追加一行 `---修改---` 然后用 1-2 句中文说明改了哪几行、原因\n"
    )
    user_parts = [
        f"【原始代码】\n{code}",
        "【失败的命令行】\n" + "\n".join(failed_lines),
        "【已成功创建的对象】\n" + (", ".join(success_objects) if success_objects else "（暂无）"),
        f"【这是第 {attempt} 轮修复】",
    ]
    if plan:
        user_parts.insert(0, "【原始绘图计划】\n" + json.dumps(plan, ensure_ascii=False))

    try:
        healed = await _call_llm(
            [{"role":"system","content":healer_sys},
             {"role":"user","content":"\n\n".join(user_parts)}],
            max_tokens=1500, temperature=0.0,
        )
    except HTTPException as e:
        return {"ok": False, "error": e.detail}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    parts = healed.split("---修改---")
    new_code = parts[0].strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""
    # 去掉可能混入的 markdown ```
    if new_code.startswith("```"):
        new_code = _re.sub(r"^```\w*\n", "", new_code)
        new_code = _re.sub(r"\n```\s*$", "", new_code)
    return {"ok": True, "code": new_code, "explanation": explanation}


@app.post("/api/academy/parse-exam", status_code=201)
async def parse_exam(
    file: UploadFile = File(...),
    title: str = Form(""),
    answer_file: Optional[UploadFile] = File(None),
):
    """上传试卷文件（图片/PDF/Word）以及可选答案解析文件，AI 解析并保存结构化题目。"""
    ext = Path(file.filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".pdf", ".docx", ".doc"}
    if ext not in allowed:
        raise HTTPException(400, f"不支持的文件类型: {ext}，请上传图片/PDF/Word")

    file_bytes = await file.read()
    if len(file_bytes) > 30 * 1024 * 1024:  # 30MB limit
        raise HTTPException(400, "文件过大，请上传 30MB 以内的文件")

    cfg = read_academy_cfg()

    # 保存原始文件
    file_id  = str(uuid.uuid4())[:12]
    saved_fn = f"{file_id}{ext}"
    (ACADEMY_FILES_DIR / saved_fn).write_bytes(file_bytes)

    # 提取内容
    extracted = await _extract_file_content(file_bytes, file.filename or "file" + ext, cfg)

    # 可选：提取答案解析文件内容
    answer_text: str = ""
    if answer_file and answer_file.filename:
        ans_ext = Path(answer_file.filename).suffix.lower()
        if ans_ext in allowed:
            ans_bytes = await answer_file.read()
            if len(ans_bytes) <= 30 * 1024 * 1024:
                ans_extracted = await _extract_file_content(ans_bytes, answer_file.filename, cfg)
                if ans_extracted["mode"] == "text":
                    answer_text = ans_extracted["content"]
                # 图片格式答案文件暂不处理（避免双视觉调用），仅记录提示

    # LLM 解析
    try:
        parsed = await _llm_parse_exam(extracted, cfg, answer_text=answer_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"解析失败: {e}") from e

    # 为每道题添加分类 + ID
    questions = parsed.get("questions", [])
    for i, q in enumerate(questions):
        q["id"] = f"{file_id}_q{i+1:03d}"
        q.setdefault("category", _classify_question(q.get("content", "")))
        q.setdefault("solution", "")
        q.setdefault("viz_suggestion", _VIZ_MAP.get(q.get("category", ""), {}))
        q.setdefault("options", [])
        q.setdefault("answer", "")
        q.setdefault("score", 0)

    exam_record = {
        "id":         file_id,
        "title":      title or parsed.get("title") or Path(file.filename or "试卷").stem,
        "subject":    parsed.get("subject", "数学"),
        "filename":   saved_fn,
        "original_name": file.filename or saved_fn,
        "file_type":  ext.lstrip("."),
        "questions":  questions,
        "question_count": len(questions),
        "created_at": datetime.now().isoformat(),
        "imported":   False,
    }

    # 上传了答案文件则立即分发到各题（不影响题目提取的完整性）
    if answer_text.strip():
        exam_record["answer_content"] = answer_text
        exam_record["answer_mode"] = "text"
        try:
            await _distribute_answers_to_questions(exam_record, answer_text)
        except Exception:
            pass

    data = read_academy_db()
    data.append(exam_record)
    write_academy_db(data)

    return {"exam": exam_record}

# ── API: 重新分类题目 ──────────────────────────────────────────

@app.post("/api/academy/exams/{exam_id}/classify")
async def classify_exam_questions(exam_id: str, body: dict = {}):
    """
    对指定试卷的题目重新分类（规则 + 可选 LLM）。
    body: { use_llm: false }
    """
    data  = read_academy_db()
    exam  = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    use_llm = body.get("use_llm", False)
    questions = exam.get("questions", [])

    # 规则分类
    for q in questions:
        cat = _classify_question(q.get("content", ""))
        q["category"] = cat
        q["viz_suggestion"] = _VIZ_MAP.get(cat, {})

    # LLM 分类剩余 "other" 题
    if use_llm:
        unclassified = [q for q in questions if q.get("category") == "other"]
        if unclassified:
            question_list = "\n".join(
                f"题号{q.get('number', i+1)}: {q.get('content', '')[:80]}"
                for i, q in enumerate(unclassified)
            )
            cat_options = "|".join(k for k in _VIZ_MAP if k != "other")
            prompt = (
                f"请为以下数学题目逐一分类，每题从以下选项中选一个：{cat_options}|other\n"
                f"严格按 JSON 返回：{{\"results\":[{{\"number\":题号,\"category\":\"分类\"}},...]}}\n\n{question_list}"
            )
            try:
                raw = await _call_llm(
                    [{"role": "user", "content": prompt}],
                    max_tokens=512,
                    response_json=True,
                )
                llm_results = json.loads(raw).get("results", [])
                num_to_cat = {str(r["number"]): r["category"] for r in llm_results}
                for q in unclassified:
                    cat = num_to_cat.get(str(q.get("number", "")), "other")
                    q["category"] = cat
                    q["viz_suggestion"] = _VIZ_MAP.get(cat, {})
            except Exception:
                pass  # LLM分类失败时保留规则分类结果

    idx = next(i for i, e in enumerate(data) if e["id"] == exam_id)
    data[idx]["questions"] = questions
    write_academy_db(data)
    return {"questions": questions}

# ── API: AI 解题 ───────────────────────────────────────────────

_SOLVE_SYSTEM = """你是专业的高中数学老师，擅长清晰解题。请为给定题目提供完整详细的解答。

要求：
1. 分步骤解答（用「第一步」「第二步」等标记），每步说明解题思路
2. 数学公式使用 LaTeX（行内用 $...$，独立行用 $$...$$）
3. 最后用「∴」或「综上」给出明确结论
4. 若有选项则明确指出选哪项
"""

@app.post("/api/academy/exams/{exam_id}/questions/{q_id}/solve")
async def solve_question(exam_id: str, q_id: str):
    """对指定题目调用 LLM 生成解题过程。"""
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    q = next((q for q in exam.get("questions", []) if q["id"] == q_id), None)
    if not q:
        raise HTTPException(404, "题目不存在")

    content = q.get("content", "")
    options_text = ""
    if q.get("options"):
        options_text = "\n选项：\n" + "\n".join(q["options"])

    user_msg = f"【题目】\n{content}{options_text}"
    if q.get("answer"):
        user_msg += f"\n\n（参考答案：{q['answer']}，请结合答案给出解题过程）"

    solution = await _call_llm(
        [{"role": "system", "content": _SOLVE_SYSTEM}, {"role": "user", "content": user_msg}],
        max_tokens=2048,
    )

    # 保存解答
    eidx = next(i for i, e in enumerate(data) if e["id"] == exam_id)
    for qobj in data[eidx]["questions"]:
        if qobj["id"] == q_id:
            qobj["solution"] = solution
            break
    write_academy_db(data)

    return {"solution": solution, "question_id": q_id}


@app.post("/api/academy/exams/{exam_id}/solve-missing")
async def solve_missing_solutions(exam_id: str):
    """批量为所有「无 solution」的题目调用 AI 生成解析。"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")

    qs = data[eidx].get("questions", [])
    targets = [q for q in qs if not (q.get("solution") or "").strip()]
    filled, failed = 0, 0
    for q in targets:
        content = q.get("content", "")
        if not content:
            continue
        options_text = ""
        if q.get("options"):
            try:
                options_text = "\n选项：\n" + "\n".join(q["options"]) if isinstance(q["options"], list) else \
                               "\n选项：\n" + "\n".join(f"{k}. {v}" for k, v in q["options"].items())
            except Exception:
                options_text = ""
        user_msg = f"【题目】\n{content}{options_text}"
        if q.get("answer"):
            user_msg += f"\n\n（参考答案：{q['answer']}，请结合答案给出解题过程）"
        try:
            solution = await _call_llm(
                [{"role": "system", "content": _SOLVE_SYSTEM},
                 {"role": "user", "content": user_msg}],
                max_tokens=2048,
            )
            q["solution"] = solution
            filled += 1
        except Exception:
            failed += 1
    write_academy_db(data)
    return {"ok": True, "filled": filled, "failed": failed, "total_missing": len(targets)}

# ── API: 可视化建议 + GeoGebra 代码生成 ───────────────────────

_VIZ_CODE_SYSTEM = """你是专业的数学可视化专家，精通 GeoGebra 脚本。
请根据题目内容生成对应的 GeoGebra 命令脚本（每行一条命令）。

规范：
- 使用 GeoGebra 命令行语法，每行一条，不加分号
- 颜色规范：主要对象蓝色(#3b82f6)、顶点琥珀(#f59e0b)、辅助线灰色(#9ca3af)
- 用 SetColor 命令设置颜色
- 必须包含 ZoomIn(-5,5,-5,5) 或 ZoomFit 控制视图范围
- 返回纯 GeoGebra 命令，不要任何解释文字
"""

@app.post("/api/academy/exams/{exam_id}/questions/{q_id}/visualize")
async def visualize_question(exam_id: str, q_id: str, body: dict = {}):
    """
    根据题目类型返回可视化建议，并可选生成 GeoGebra 代码。
    body: { generate_code: true/false }
    """
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    q = next((q for q in exam.get("questions", []) if q["id"] == q_id), None)
    if not q:
        raise HTTPException(404, "题目不存在")

    category = q.get("category", "other")
    viz_suggestion = _VIZ_MAP.get(category, {"viz_type": "none", "engine": "none", "desc": "暂无可视化建议"})

    result: dict = {
        "question_id": q_id,
        "category": category,
        "viz_suggestion": viz_suggestion,
        "geogebra_code": None,
    }

    generate_code = body.get("generate_code", False)
    if generate_code and viz_suggestion.get("engine") == "geogebra":
        content = q.get("content", "")
        solution = q.get("solution", "")
        user_msg = f"【题目类型】{category}\n【题目】{content}"
        if solution:
            user_msg += f"\n【解题过程摘要】{solution[:300]}"
        try:
            code = await _call_llm(
                [{"role": "system", "content": _VIZ_CODE_SYSTEM}, {"role": "user", "content": user_msg}],
                max_tokens=1024,
            )
            result["geogebra_code"] = code.strip()
            # 保存到题目
            eidx = next(i for i, e in enumerate(data) if e["id"] == exam_id)
            for qobj in data[eidx]["questions"]:
                if qobj["id"] == q_id:
                    qobj["viz_suggestion"] = viz_suggestion
                    qobj["geogebra_code"] = code.strip()
                    break
            write_academy_db(data)
        except HTTPException:
            raise
        except Exception as e:
            result["error"] = str(e)

    return result

# ── API: 导入到数据集 ──────────────────────────────────────────

@app.post("/api/academy/exams/{exam_id}/import")
def import_exam_to_dataset(exam_id: str, body: dict = {}):
    """
    将试卷中选中（或全部）题目导入到数据标注数据集。
    body: { question_ids: ["id1", "id2"] }  // 空列表 = 全部
    """
    exam_data = read_academy_db()
    exam = next((e for e in exam_data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    q_ids = body.get("question_ids", [])
    questions = exam.get("questions", [])
    if q_ids:
        questions = [q for q in questions if q["id"] in q_ids]

    if not questions:
        raise HTTPException(400, "没有可导入的题目")

    dataset = read_db()
    imported_ids = []
    now = datetime.now().isoformat()
    for q in questions:
        seq = next_seq(dataset)
        tid = f"q{seq:03d}"
        triplet: dict = {
            "id": tid, "seq": seq,
            "category": q.get("category", "other"),
            "dim": "3d" if q.get("category") == "solid_geometry" else "2d",
            "question_text": q.get("content", ""),
            "question_latex": q.get("content", ""),
            "solution_text":  q.get("solution", ""),
            "solution_latex": q.get("solution", ""),
            "images": [],
            "codes": [],
            "style_features": StyleFeatures().model_dump(),
            "annotation_status": "draft",
            "annotator": "智绘书院",
            "notes": f"从试卷「{exam['title']}」题目{q.get('number','')}导入",
            "created_at": now,
            "updated_at": now,
        }
        # 若已生成 GeoGebra 代码，一并导入
        gb_code = q.get("geogebra_code", "")
        if gb_code:
            triplet["codes"].append({
                "id": str(uuid.uuid4())[:8],
                "engine": "geogebra",
                "dim": triplet["dim"],
                "code": gb_code,
                "description": f"智绘书院生成（题目{q.get('number','')}）",
                "image_refs": [],
                "is_normalized": False,
            })
            triplet["style_features"] = auto_detect_style_features(gb_code, "geogebra")
        dataset.append(triplet)
        imported_ids.append(tid)

    write_db(dataset)

    # 标记该试卷已导入
    eidx = next(i for i, e in enumerate(exam_data) if e["id"] == exam_id)
    exam_data[eidx]["imported"] = True
    write_academy_db(exam_data)

    return {"ok": True, "imported_count": len(imported_ids), "triplet_ids": imported_ids}

# ── API: 试卷列表 / 详情 / 删除 ───────────────────────────────

@app.get("/api/academy/exams")
def list_academy_exams():
    data = read_academy_db()
    # 返回摘要（不含题目详情以节省带宽）
    summaries = [{
        "id": e["id"],
        "title": e["title"],
        "subject": e.get("subject", "数学"),
        "file_type": e.get("file_type", ""),
        "original_name": e.get("original_name", ""),
        "question_count": e.get("question_count", 0),
        "created_at": e.get("created_at", ""),
        "imported": e.get("imported", False),
        "published": e.get("published", False),
        "allow_view_answer": e.get("allow_view_answer", False),
    } for e in data]
    return {"exams": summaries, "total": len(summaries)}

@app.get("/api/academy/exams/{exam_id}")
def get_academy_exam(exam_id: str):
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    return {"exam": exam}

@app.delete("/api/academy/exams/{exam_id}")
def delete_academy_exam(exam_id: str):
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    # 删除原始文件
    fp = ACADEMY_FILES_DIR / exam.get("filename", "")
    if fp.exists():
        fp.unlink()
    data = [e for e in data if e["id"] != exam_id]
    write_academy_db(data)
    return {"ok": True}

@app.put("/api/academy/exams/{exam_id}")
def update_academy_exam(exam_id: str, body: dict):
    """更新试卷元信息（目前仅支持改名 title）。"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    allowed = {"title"}
    for k, v in (body or {}).items():
        if k in allowed and isinstance(v, str) and v.strip():
            data[eidx][k] = v.strip()
    data[eidx]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_academy_db(data)
    return {"ok": True, "exam": {k: data[eidx].get(k) for k in ("id", "title", "updated_at")}}

@app.put("/api/academy/exams/{exam_id}/questions/{q_id}")
def update_academy_question(exam_id: str, q_id: str, body: dict):
    """更新题目字段（content / answer / solution / category / geogebra_code / viz_engine 等）。"""
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    qs = exam.get("questions", []) or []
    q = next((x for x in qs if str(x.get("id")) == str(q_id) or str(x.get("number")) == str(q_id)), None)
    if not q:
        raise HTTPException(404, "题目不存在")
    allowed = {
        "content", "answer", "solution", "category",
        "geogebra_code", "viz_engine", "options", "number",
    }
    for k, v in (body or {}).items():
        if k in allowed:
            q[k] = v
    exam["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_academy_db(data)
    return {"ok": True, "question": q}


# ════════════════════════════════════════════════════════════
# 🏭 出题工厂（LangGraph Multi-Agent）
# ════════════════════════════════════════════════════════════

@app.post("/api/academy/exams/{exam_id}/questions/{q_id}/spawn-variants")
async def spawn_variants(exam_id: str, q_id: str, body: dict = {}):
    """流式生成变式题（SSE）。
    body: { n: 3 }
    返回事件流，每行 'data: {...json...}\\n\\n'。
    """
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    q = next((x for x in (exam.get("questions") or [])
              if str(x.get("id")) == str(q_id) or str(x.get("number")) == str(q_id)), None)
    if not q:
        raise HTTPException(404, "题目不存在")

    n = max(1, min(5, int(body.get("n") or 3)))
    src = {
        "content":       q.get("content", ""),
        "answer":        q.get("answer", ""),
        "solution":      q.get("solution", ""),
        "options":       q.get("options", []),
        "category":      q.get("category", ""),
        "viz_engine":    q.get("viz_engine", ""),
        "geogebra_code": q.get("geogebra_code", ""),
    }

    from agent_factory import run_factory_stream

    async def gen():
        try:
            async for ev in run_factory_stream(src, n=n):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/academy/exams/{exam_id}/accept-variants")
def accept_variants(exam_id: str, body: dict):
    """教师审核通过后，把选中的变式题作为新题目追加到本试卷末尾。
    body: { source_qid: str, variants: [{content, answer, solution, options, geogebra_code, viz_engine, difficulty, strategy}] }
    """
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    exam = data[eidx]
    src_qid = body.get("source_qid", "")
    variants = body.get("variants") or []
    if not variants:
        raise HTTPException(400, "未选择任何变式题")

    qs = exam.get("questions") or []
    # 找到原题，定位插入位置（紧跟其后）
    insert_at = len(qs)
    for i, x in enumerate(qs):
        if str(x.get("id")) == str(src_qid) or str(x.get("number")) == str(src_qid):
            insert_at = i + 1
            break
    base_no = max([int(x.get("number") or 0) for x in qs] + [0])
    new_items = []
    for i, v in enumerate(variants):
        base_no += 1
        new_items.append({
            "id":            str(uuid.uuid4())[:12],
            "number":        base_no,
            "content":       v.get("content", ""),
            "answer":        v.get("answer", ""),
            "solution":      v.get("solution", ""),
            "options":       v.get("options", []),
            "category":      v.get("category", ""),
            "geogebra_code": v.get("geogebra_code", ""),
            "viz_engine":    v.get("viz_engine", ""),
            "is_variant":    True,
            "source_qid":    src_qid,
            "variant_meta":  {
                "strategy":   v.get("strategy"),
                "difficulty": v.get("difficulty"),
                "focus":      v.get("focus"),
            },
            "created_at":    datetime.now().isoformat(timespec="seconds"),
        })
    qs[insert_at:insert_at] = new_items
    exam["questions"] = qs
    exam["question_count"] = len(qs)
    exam["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_academy_db(data)
    return {"ok": True, "added": len(new_items), "first_id": new_items[0]["id"] if new_items else None}


# ── 变式题草稿（未入库的暂存）────────────────────────────────
# 存储位置：每道源题节点上的 "variant_drafts": [v, v, ...]
# 这样草稿跟随源题，删源题草稿一并消失，无需独立表。

@app.get("/api/academy/exams/{exam_id}/questions/{q_id}/variant-drafts")
def get_variant_drafts(exam_id: str, q_id: str):
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    q = next((x for x in (exam.get("questions") or [])
              if str(x.get("id")) == str(q_id) or str(x.get("number")) == str(q_id)), None)
    if not q:
        raise HTTPException(404, "题目不存在")
    return {"drafts": q.get("variant_drafts") or [], "saved_at": q.get("variant_drafts_saved_at", "")}


@app.put("/api/academy/exams/{exam_id}/questions/{q_id}/variant-drafts")
def save_variant_drafts(exam_id: str, q_id: str, body: dict):
    """把当前生成结果（含编辑过的）暂存到源题节点。
    body: { variants: [...] }
    """
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    exam = data[eidx]
    qs = exam.get("questions") or []
    qidx = next(
        (i for i, x in enumerate(qs)
         if str(x.get("id")) == str(q_id) or str(x.get("number")) == str(q_id)),
        None,
    )
    if qidx is None:
        raise HTTPException(404, "题目不存在")
    drafts = body.get("variants") or []
    qs[qidx]["variant_drafts"] = drafts
    qs[qidx]["variant_drafts_saved_at"] = datetime.now().isoformat(timespec="seconds")
    exam["questions"] = qs
    exam["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_academy_db(data)
    return {"ok": True, "count": len(drafts)}


@app.delete("/api/academy/exams/{exam_id}/questions/{q_id}/variant-drafts")
def clear_variant_drafts(exam_id: str, q_id: str):
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    exam = data[eidx]
    qs = exam.get("questions") or []
    qidx = next(
        (i for i, x in enumerate(qs)
         if str(x.get("id")) == str(q_id) or str(x.get("number")) == str(q_id)),
        None,
    )
    if qidx is None:
        raise HTTPException(404, "题目不存在")
    qs[qidx].pop("variant_drafts", None)
    qs[qidx].pop("variant_drafts_saved_at", None)
    exam["questions"] = qs
    write_academy_db(data)
    return {"ok": True}


# ── API: 批量解题 / 可视化（后台任务风格，简单版）────────────

@app.post("/api/academy/exams/{exam_id}/batch-solve")
async def batch_solve_exam(exam_id: str, body: dict = {}):
    """批量为试卷中所有未解题的题目生成解答（注意：同步串行，可能耗时）。"""
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    questions = exam.get("questions", [])
    limit = body.get("limit", 5)  # 每次最多解 N 道，防止超时

    solved = []
    eidx = next(i for i, e in enumerate(data) if e["id"] == exam_id)
    for qobj in questions:
        if qobj.get("solution"):
            continue
        if len(solved) >= limit:
            break
        content = qobj.get("content", "")
        options_text = ""
        if qobj.get("options"):
            options_text = "\n选项：\n" + "\n".join(qobj["options"])
        user_msg = f"【题目】\n{content}{options_text}"
        try:
            solution = await _call_llm(
                [{"role": "system", "content": _SOLVE_SYSTEM}, {"role": "user", "content": user_msg}],
                max_tokens=1500,
            )
            qobj["solution"] = solution
            solved.append(qobj["id"])
        except Exception:
            break  # LLM错误时停止批量，避免刷屏错误

    data[eidx]["questions"] = questions
    write_academy_db(data)
    return {"solved_count": len(solved), "solved_ids": solved}


# ══════════════════════════════════════════════════════════════
#  智绘书院 — 用户/角色系统（admin / teacher / student）
# ══════════════════════════════════════════════════════════════

ACADEMY_USERS_FILE = BASE_DIR / "academy_users.json"

_DEFAULT_USERS = [
    {"id": "admin001",   "username": "admin",   "password": "admin123",   "role": "admin",   "name": "管理员",  "created_at": ""},
    {"id": "teacher001", "username": "teacher", "password": "teacher123", "role": "teacher", "name": "教师",    "created_at": ""},
    {"id": "student001", "username": "student", "password": "student123", "role": "student", "name": "学生",    "created_at": ""},
]

def read_users() -> List[dict]:
    if not ACADEMY_USERS_FILE.exists():
        data = [{**u, "created_at": datetime.now().isoformat()} for u in _DEFAULT_USERS]
        ACADEMY_USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    text = ACADEMY_USERS_FILE.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []

def write_users(data: List[dict]) -> None:
    tmp = ACADEMY_USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, ACADEMY_USERS_FILE)

@app.post("/api/academy/login")
def academy_login(body: dict):
    """
    简单密码登录。
    body: { username, password }
    返回: { ok, user: {id, username, role, name} }
    """
    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(400, "用户名和密码不能为空")
    users = read_users()
    user = next((u for u in users if u["username"] == username and u["password"] == password), None)
    if not user:
        raise HTTPException(401, "用户名或密码错误")
    return {"ok": True, "user": {k: user[k] for k in ("id", "username", "role", "name")}}

@app.get("/api/academy/users")
def list_academy_users():
    users = read_users()
    return {"users": [{k: u[k] for k in ("id", "username", "role", "name", "created_at")} for u in users]}

@app.post("/api/academy/users", status_code=201)
def create_academy_user(body: dict):
    """管理员创建用户。body: { username, password, role, name }"""
    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    role     = (body.get("role") or "student").strip()
    name     = (body.get("name") or username).strip()
    if not username or not password:
        raise HTTPException(400, "用户名和密码不能为空")
    if role not in ("admin", "teacher", "student"):
        raise HTTPException(400, "角色必须为 admin/teacher/student")
    users = read_users()
    if any(u["username"] == username for u in users):
        raise HTTPException(409, f"用户名 {username} 已存在")
    new_user = {
        "id": str(uuid.uuid4())[:10],
        "username": username,
        "password": password,
        "role": role,
        "name": name,
        "created_at": datetime.now().isoformat(),
    }
    users.append(new_user)
    write_users(users)
    return {"ok": True, "user": {k: new_user[k] for k in ("id", "username", "role", "name")}}

@app.delete("/api/academy/users/{user_id}")
def delete_academy_user(user_id: str):
    users = read_users()
    users = [u for u in users if u["id"] != user_id]
    write_users(users)
    return {"ok": True}

@app.put("/api/academy/users/{user_id}/password")
def change_user_password(user_id: str, body: dict):
    """修改密码。body: { new_password }"""
    new_pw = (body.get("new_password") or "").strip()
    if not new_pw:
        raise HTTPException(400, "新密码不能为空")
    users = read_users()
    idx = next((i for i, u in enumerate(users) if u["id"] == user_id), None)
    if idx is None:
        raise HTTPException(404, "用户不存在")
    users[idx]["password"] = new_pw
    write_users(users)
    return {"ok": True}


# ── 试卷发布 ───────────────────────────────────────────────────
# 发布后学生可见；可关联一份答案文件

@app.post("/api/academy/exams/{exam_id}/publish")
async def publish_exam(
    exam_id: str,
    answer_file: UploadFile = File(None),
    answer_content: str = Form(""),
    allow_view_answer: str = Form("false"),
):
    """
    教师发布试卷（可同时上传答案文件，或直接粘贴纯文本答案）。
    - 将 published=True，记录 published_at
    - 若提供答案文件，AI 解析答案内容并存入 exam.answer_content
    - 若仅提供 answer_content 纯文本，则直接保存
    - allow_view_answer：是否允许学生查看答案/解析（默认 false，可随后切换）
    """
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")

    eidx = next(i for i, e in enumerate(data) if e["id"] == exam_id)

    # 0) 若仅提供纯文本答案
    if (not answer_file or not answer_file.filename) and answer_content.strip():
        data[eidx]["answer_content"] = answer_content.strip()[:8000]
        data[eidx]["answer_mode"] = "text"

    # 处理答案文件
    if answer_file and answer_file.filename:
        ext = Path(answer_file.filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".docx", ".doc", ".txt"}
        if ext not in allowed:
            raise HTTPException(400, f"不支持的答案文件格式: {ext}")
        ans_bytes = await answer_file.read()
        if len(ans_bytes) > 20 * 1024 * 1024:
            raise HTTPException(400, "答案文件过大（限20MB）")

        # 保存文件
        ans_fn = f"{exam_id}_ans{ext}"
        (ACADEMY_FILES_DIR / ans_fn).write_bytes(ans_bytes)
        data[eidx]["answer_filename"] = ans_fn
        data[eidx]["answer_original_name"] = answer_file.filename

        # 若是文本/PDF/Word 则提取答案内容
        cfg = read_academy_cfg()
        if ext == ".txt":
            try:
                ans_text = ans_bytes.decode("utf-8", errors="ignore")
                data[eidx]["answer_content"] = ans_text[:8000]
                data[eidx]["answer_mode"] = "text"
            except Exception:
                pass
        else:
            extracted = await _extract_file_content(ans_bytes, answer_file.filename, cfg)
            if extracted["mode"] == "text":
                data[eidx]["answer_content"] = extracted["content"]
                data[eidx]["answer_mode"] = "text"
            else:
                # 图片答案：尝试用视觉模型提取文字答案
                try:
                    ans_text = await _call_llm(
                        [
                            {"role": "system", "content": "你是专业的数学老师。请从答案图片中提取每道题的答案，格式：第N题：答案内容"},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": extracted["content"]}},
                                {"type": "text", "text": "请提取图片中所有题目的答案"},
                            ]},
                        ],
                        model=extracted.get("vision_model", ""),
                        max_tokens=2048,
                    )
                    data[eidx]["answer_content"] = ans_text
                    data[eidx]["answer_mode"] = "text"
                except Exception:
                    # 存图片内容（base64）以备前端展示
                    data[eidx]["answer_content"] = extracted["content"]
                    data[eidx]["answer_mode"] = "image_b64"

    data[eidx]["published"] = True
    data[eidx]["published_at"] = datetime.now().isoformat()
    # 为每道题打上 published_at（仅首次发布时；用于"待重新发布"判定）
    _now_iso = data[eidx]["published_at"]
    for _q in data[eidx].get("questions", []):
        if not _q.get("published_at"):
            _q["published_at"] = _now_iso
            _q["version"] = int(_q.get("version", 0)) or 1
        _q["has_pending_changes"] = False
        _q["pending_fields"] = []
    # 允许学生查看答案/解析（默认 False；接受 "true"/"1"/"on"/"yes"）
    _flag = str(allow_view_answer or "").strip().lower()
    data[eidx]["allow_view_answer"] = _flag in ("true", "1", "on", "yes")

    # 将整卷答案文本切分到各小题（仅当解析得到了纯文本答案时）
    distrib_summary = {"distributed": 0, "skipped": 0}
    ans_text_for_split = data[eidx].get("answer_content", "") if data[eidx].get("answer_mode") == "text" else ""
    if ans_text_for_split and data[eidx].get("questions"):
        try:
            distrib_summary = await _distribute_answers_to_questions(data[eidx], ans_text_for_split)
        except Exception as _e:
            distrib_summary = {"distributed": 0, "skipped": 0, "error": str(_e)}

    write_academy_db(data)
    return {"ok": True, "exam": {k: data[eidx].get(k) for k in
        ("id", "title", "published", "published_at", "allow_view_answer", "answer_filename", "answer_original_name")},
        "answer_distribution": distrib_summary}

@app.post("/api/academy/exams/{exam_id}/answer-visibility")
async def set_answer_visibility(exam_id: str, body: dict):
    """教师随时切换学生是否可查看答案/解析。body: {allow: bool}
    若 body 包含 apply_to_questions=True，则同步覆盖每题的 allow_view_answer。"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    allow = bool(body.get("allow"))
    data[eidx]["allow_view_answer"] = allow
    if body.get("apply_to_questions"):
        for q in data[eidx].get("questions", []):
            q["allow_view_answer"] = allow
    write_academy_db(data)
    return {"ok": True, "exam_id": exam_id, "allow_view_answer": allow}

@app.post("/api/academy/exams/{exam_id}/questions/{qid}/answer-visibility")
async def set_question_answer_visibility(exam_id: str, qid: str, body: dict):
    """教师按单题切换学生是否可查看该题答案/解析。body: {allow: bool}"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    qidx = next((i for i, q in enumerate(data[eidx].get("questions", []))
                 if str(q.get("id")) == str(qid) or str(q.get("number")) == str(qid)), None)
    if qidx is None:
        raise HTTPException(404, "题目不存在")
    allow = bool(body.get("allow"))
    data[eidx]["questions"][qidx]["allow_view_answer"] = allow
    write_academy_db(data)
    return {"ok": True, "exam_id": exam_id, "qid": qid, "allow_view_answer": allow}

def _question_answer_allowed(exam: dict, q: dict) -> bool:
    """单题答案/解析对学生是否可见。优先取题目级，缺省回落到试卷级。"""
    if "allow_view_answer" in q:
        return bool(q.get("allow_view_answer"))
    return bool(exam.get("allow_view_answer"))

@app.post("/api/academy/exams/{exam_id}/unpublish")
def unpublish_exam(exam_id: str):
    """取消发布（撤回，学生不可见）。"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    data[eidx]["published"] = False
    write_academy_db(data)
    return {"ok": True}


@app.post("/api/academy/exams/{exam_id}/redistribute-answers")
async def redistribute_answers(exam_id: str, body: dict | None = None):
    """重新把整卷答案文本切分到各小题。
    body: {overwrite: bool}，overwrite=True 时会覆盖已有 answer/solution。"""
    data = read_academy_db()
    eidx = next((i for i, e in enumerate(data) if e["id"] == exam_id), None)
    if eidx is None:
        raise HTTPException(404, "试卷不存在")
    exam = data[eidx]
    text = exam.get("answer_content", "")
    if not text or exam.get("answer_mode") != "text":
        raise HTTPException(400, "尚未上传/解析出文本答案，无法分发")

    overwrite = bool((body or {}).get("overwrite"))
    if overwrite:
        for q in exam.get("questions", []):
            q.pop("answer", None)
            q.pop("solution", None)

    summary = await _distribute_answers_to_questions(exam, text)
    write_academy_db(data)
    return {"ok": True, **summary}


# ── 学生端：查看已发布试卷 ─────────────────────────────────────

@app.get("/api/academy/student/exams")
def student_list_exams():
    """学生获取所有已发布试卷列表（不含答案）。"""
    data = read_academy_db()
    result = []
    for e in data:
        if not e.get("published"):
            continue
        result.append({
            "id": e["id"],
            "title": e["title"],
            "subject": e.get("subject", "数学"),
            "question_count": e.get("question_count", 0),
            "published_at": e.get("published_at", ""),
            "has_answer": bool(e.get("answer_content") or e.get("answer_filename")),
            "has_solutions": any(q.get("solution") for q in e.get("questions", [])),
        })
    return {"exams": result, "total": len(result)}

@app.get("/api/academy/student/exams/{exam_id}")
def student_get_exam(exam_id: str):
    """学生查看已发布试卷题目（不含答案，答案需单独请求）。"""
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    if not exam.get("published"):
        raise HTTPException(403, "该试卷尚未发布")
    allow_answer = bool(exam.get("allow_view_answer"))
    # 返回题目，不含答案内容
    qs = []
    for q in exam.get("questions", []):
        q_allow = _question_answer_allowed(exam, q)
        qs.append({
            "id": q["id"],
            "number": q.get("number"),
            "type": q.get("type"),
            "content": q.get("content", ""),
            "options": q.get("options", []),
            "category": q.get("category", ""),
            "score": q.get("score", 0),
            "allow_view_answer": q_allow,
            # 单题级开关：仅在该题对学生开放时才告知“有解析”
            "has_solution": bool(q.get("solution")) and q_allow,
            "geogebra_code": q.get("geogebra_code", ""),
            "viz_engine": q.get("viz_engine", ""),
            "is_variant": bool(q.get("is_variant")),
            "source_qid": q.get("source_qid", ""),
            "variant_meta": q.get("variant_meta", {}),
        })
    return {
        "exam": {
            "id": exam["id"],
            "title": exam["title"],
            "subject": exam.get("subject", "数学"),
            "published_at": exam.get("published_at", ""),
            "question_count": exam.get("question_count", 0),
            "allow_view_answer": allow_answer,
            "has_answer": bool(exam.get("answer_content") or exam.get("answer_filename")) and allow_answer,
            "questions": qs,
        }
    }

@app.get("/api/academy/student/exams/{exam_id}/answers")
def student_get_answers(exam_id: str):
    """学生查看已发布试卷的答案（需教师开启 allow_view_answer）。"""
    data = read_academy_db()
    exam = next((e for e in data if e["id"] == exam_id), None)
    if not exam:
        raise HTTPException(404, "试卷不存在")
    if not exam.get("published"):
        raise HTTPException(403, "该试卷尚未发布")

    # 每题答案 + AI解答（按题级开关过滤）
    q_answers = []
    any_open = False
    for q in exam.get("questions", []):
        q_allow = _question_answer_allowed(exam, q)
        if not q_allow:
            # 跳过未开放的题目，前端不会渲染
            continue
        any_open = True
        q_answers.append({
            "id": q["id"],
            "number": q.get("number"),
            "answer": q.get("answer", ""),
            "solution": q.get("solution", ""),
            "geogebra_code": q.get("geogebra_code", ""),
            "viz_engine": q.get("viz_engine", ""),
            "viz_suggestion": q.get("viz_suggestion", {}),
        })
    if not any_open and not exam.get("allow_view_answer"):
        raise HTTPException(403, "教师未开放任何题目的答案查看权限")

    return {
        "exam_id": exam_id,
        "title": exam["title"],
        "answer_content": exam.get("answer_content", ""),
        "answer_mode": exam.get("answer_mode", "text"),
        "answer_original_name": exam.get("answer_original_name", ""),
        "published_at": exam.get("published_at", ""),
        "question_answers": q_answers,
    }


# ── 单独路由：三端页面 ─────────────────────────────────────────

@app.get("/academy-student.html", response_class=HTMLResponse)
def serve_academy_student():
    html_path = Path(__file__).parent / "academy-student.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy-student.html 未找到</h1>", status_code=404)

@app.get("/academy-admin.html", response_class=HTMLResponse)
def serve_academy_admin():
    html_path = Path(__file__).parent / "academy-admin.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>academy-admin.html 未找到</h1>", status_code=404)


@app.get("/api/wolfram")
async def wolfram_proxy(q: str = ""):
    """
    Wolfram Alpha Short Answers API proxy (avoids CORS).
    Frontend calls: GET /api/wolfram?q=<query>
    Requires WOLFRAM_APP_ID env var (free account at developer.wolframalpha.com)
    """
    app_id = os.environ.get("WOLFRAM_APP_ID", "")
    if not app_id:
        return {"error": "WOLFRAM_APP_ID env var not set", "result": ""}
    try:
        url = f"https://api.wolframalpha.com/v1/result?i={q}&appid={app_id}&units=metric"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            return {"result": r.text, "status": r.status_code}
    except Exception as e:
        return {"error": str(e), "result": ""}

# ── 入口 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  数学可视化数据集标注系统")
    print("  数据存储: F:/dataset-annotator/data/")
    print("  访问地址: http://localhost:8765")
    print("=" * 55)
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True, reload_dirs=["."])
