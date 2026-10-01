"""FastAPI application for industrial surface anomaly localization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .imaging import MAX_UPLOAD_BYTES, SUPPORTED_FORMATS
from .jobs import JobStore
from .store import AppState
from .video_pipeline import (
    MAX_VIDEO_BYTES,
    SUPPORTED_VIDEO_FORMATS,
    VIDEO_PREPROCESS_INFO,
    VideoParams,
    suggest_params,
    validate_video_file,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
STATIC_DIR = Path(__file__).resolve().parent / "static"
EXAMPLES_DIR = ROOT / "examples"

app = FastAPI(title="Industrial Surface Anomaly Inspector", version="0.1.0")
state = AppState(DATA_DIR, WEIGHTS)
job_store = JobStore(DATA_DIR, state)
job_store.recover_stale()

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


# ----------------------------------------------------------- conveyor video

def _parse_int4(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="roi 必须是 x1,y1,x2,y2") from exc
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="roi 必须包含 4 个整数")
    return tuple(parts)


@app.get("/api/video/limits")
def video_limits() -> dict:
    return VIDEO_PREPROCESS_INFO


@app.post("/api/video/validate")
async def video_validate(
    video: UploadFile = File(...), background: UploadFile = File(...)
) -> dict:
    import io
    import uuid
    from PIL import Image

    video_bytes = await _read_video(video)
    bg_bytes = await background.read()
    tmp_dir = DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    vp = tmp_dir / f"video_{token}.bin"
    bp = tmp_dir / f"bg_{token}.png"
    vp.write_bytes(video_bytes)
    try:
        from .imaging import decode_image
        pil = decode_image(bg_bytes)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        bp.write_bytes(buf.getvalue())
        info = validate_video_file(str(vp), str(bp))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if vp.exists():
            vp.unlink()
        if bp.exists():
            bp.unlink()
    info["suggested"] = suggest_params(info["width"], info["height"])
    return info


async def _read_video(video: UploadFile) -> bytes:
    suffix = (video.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"仅支持 {'/'.join(f.upper() for f in SUPPORTED_VIDEO_FORMATS)} 视频",
        )
    data = await video.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"视频超过 {MAX_VIDEO_BYTES // (1024 * 1024)} MB 上限",
        )
    return data


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    background: UploadFile = File(...),
    roi: str = Form(...),
    line_x: int = Form(...),
    direction: str = Form(...),
    diff_threshold: int = Form(32),
    min_area: int = Form(400),
    max_piece_area: int = Form(20000),
    min_side: int = Form(18),
    max_side: int = Form(160),
) -> dict:
    video_bytes = await _read_video(video)
    bg_bytes = await background.read()
    try:
        from .imaging import decode_image
        decode_image(bg_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    params = VideoParams(
        roi=_parse_int4(roi),
        line_x=line_x,
        direction=direction,
        diff_threshold=diff_threshold,
        min_area=min_area,
        max_piece_area=max_piece_area,
        min_side=min_side,
        max_side=max_side,
    )
    import uuid
    job_id = uuid.uuid4().hex[:12]
    try:
        return job_store.create(
            job_id,
            video.filename or "video.mp4",
            video_bytes,
            bg_bytes,
            params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": job_store.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return job_store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在")


@app.get("/api/jobs/{job_id}/progress")
def get_job_progress(job_id: str) -> dict:
    try:
        return job_store.progress(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        return job_store.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> Response:
    try:
        path = job_store.video_path(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="任务不存在")
    return FileResponse(
        str(path), media_type="video/mp4"
    )


@app.get("/api/jobs/{job_id}/evidence/{track_id}/{kind}.png")
def job_evidence(job_id: str, track_id: int, kind: str) -> Response:
    if kind not in ("crop", "heat"):
        raise HTTPException(status_code=404, detail="unknown evidence kind")
    try:
        data = job_store.evidence_png(job_id, track_id, kind)
    except KeyError:
        raise HTTPException(status_code=404, detail="证据不存在（可能待复核）")
    return Response(content=data, media_type="image/png")
