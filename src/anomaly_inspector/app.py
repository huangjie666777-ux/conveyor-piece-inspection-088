"""FastAPI application for industrial surface anomaly localization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .imaging import MAX_UPLOAD_BYTES, SUPPORTED_FORMATS
from .store import AppState

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = ROOT / "examples"

app = FastAPI(title="Industrial Surface Anomaly Inspector", version="0.1.0")
state = AppState(DATA_DIR, WEIGHTS)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")


@app.get("/")
def index() -> Response:
    return Response(
        (STATIC_DIR / "index.html").read_text(), media_type="text/html"
    )


@app.get("/api/status")
def status() -> dict:
    return state.status()


@app.get("/api/samples/{group}")
def list_samples(group: str) -> dict:
    _validate_group(group)
    return {
        "samples": [
            {
                "id": s.sample_id,
                "filename": s.filename,
                "width": s.width,
                "height": s.height,
            }
            for s in state.list_samples(group)
        ]
    }


@app.get("/api/samples/{group}/{sample_id}/image")
def sample_image(group: str, sample_id: str) -> Response:
    _validate_group(group)
    try:
        return Response(content=state.sample_png(sample_id), media_type="image/png")
    except KeyError:
        raise HTTPException(status_code=404, detail="sample not found")


@app.post("/api/samples/{group}")
async def upload_sample(group: str, file: UploadFile = File(...)) -> dict:
    _validate_group(group)
    if (file.content_type or "").lower() not in (
        "image/png",
        "image/jpeg",
        "image/jpg",
    ):
        raise HTTPException(
            status_code=415,
            detail="only PNG or JPEG uploads are accepted",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        sample = state.add_sample(group, file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": sample.sample_id, "filename": sample.filename}


@app.delete("/api/samples/{group}/{sample_id}")
def delete_sample(group: str, sample_id: str) -> dict:
    _validate_group(group)
    try:
        state.delete_sample(sample_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="sample not found")
    return {"deleted": sample_id}


@app.post("/api/build")
def build_bank() -> dict:
    try:
        return state.rebuild()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)) -> dict:
    if (file.content_type or "").lower() not in (
        "image/png",
        "image/jpeg",
        "image/jpg",
    ):
        raise HTTPException(
            status_code=415, detail="only PNG or JPEG uploads are accepted"
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds 10 MB limit")
    try:
        result = state.inspect_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    import base64

    return {
        "score": result["score"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "threshold_is_zero": result["threshold_is_zero"],
        "color_scale_vmax": result["color_scale_vmax"],
        "width": result["width"],
        "height": result["height"],
        "original": base64.b64encode(result["original_png"]).decode(),
        "heatmap": base64.b64encode(result["heat_png"]).decode(),
    }


def _validate_group(group: str) -> None:
    if group not in ("reference", "calibration"):
        raise HTTPException(status_code=404, detail="unknown sample group")
