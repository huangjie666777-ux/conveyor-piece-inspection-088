"""FastAPI application for industrial surface anomaly localization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .imaging import MAX_UPLOAD_BYTES, SUPPORTED_FORMATS
from .jobs import JobError, JobManager
from .detection import DetectionParams
from .video import (
    MAX_VIDEO_BYTES,
    VIDEO_EXTENSIONS,
    VideoConfig,
)
from .store import AppState

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = ROOT / "examples"

app = FastAPI(title="Industrial Surface Anomaly Inspector", version="0.1.0")
state = AppState(DATA_DIR, WEIGHTS)
jobs = JobManager(DATA_DIR / "jobs", state)

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


# ------------------------------------------------------------- conveyor video

@app.get("/api/video/limits")
def video_limits() -> dict:
    return {
        "video_extensions": list(VIDEO_EXTENSIONS),
        "video_max_bytes": MAX_VIDEO_BYTES,
        "background_formats": list(SUPPORTED_FORMATS),
        "background_max_bytes": MAX_UPLOAD_BYTES,
        "fps_range": [1.0, 120.0],
        "frame_side_range": [120, 3840],
        "note": "背景图必须与视频同尺寸；视频逐帧解码，不会整体载入内存",
    }


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": jobs.list_jobs()}


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    background: UploadFile = File(...),
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_w: int = Form(...),
    roi_h: int = Form(...),
    line_x: int = Form(...),
    direction: str = Form(...),
) -> dict:
    video_bytes = await video.read()
    bg_bytes = await background.read()
    if (background.content_type or "").lower() not in (
        "image/png",
        "image/jpeg",
        "image/jpg",
    ):
        raise HTTPException(415, "背景图仅支持 PNG/JPEG")
    if direction not in ("lr", "rl"):
        raise HTTPException(400, "direction 必须是 lr 或 rl")
    config = VideoConfig(
        roi=(roi_x, roi_y, roi_w, roi_h),
        line_x=line_x,
        direction=direction,
        params=DetectionParams(),
    )
    try:
        return jobs.create_job(
            video.filename or "video.mp4", video_bytes, bg_bytes, config
        )
    except JobError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        return jobs.status(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        return jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    try:
        paths = jobs.paths(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    return FileResponse(paths["video"])  # supports HTTP range requests


@app.get("/api/jobs/{job_id}/evidence/{track_id}/{kind}")
def job_evidence(job_id: str, track_id: str, kind: str) -> Response:
    if kind not in ("crop", "heatmap"):
        raise HTTPException(404, "unknown evidence kind")
    if not track_id.isdigit():
        raise HTTPException(400, "bad track id")
    try:
        paths = jobs.paths(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    path = paths["dir"] / "evidence" / f"{track_id}_{kind}.png"
    if not path.exists():
        raise HTTPException(404, "该工件缺少可靠证据（待复核）")
    return Response(content=path.read_bytes(), media_type="image/png")
