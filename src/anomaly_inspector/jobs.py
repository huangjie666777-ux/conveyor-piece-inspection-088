"""Persistent background video job storage and worker threads."""

from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import imaging
from .store import AppState
from .video_pipeline import (
    CancellationToken,
    Progress,
    VideoParams,
    process_video,
    validate_video_file,
)


class JobStore:
    def __init__(self, data_dir: Path, state: AppState):
        self.root = Path(data_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = state
        self._lock = threading.RLock()
        self.active: dict[str, CancellationToken] = {}
        self._thread: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------- queries

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = []
            for meta_path in sorted(self.root.glob("*/job.json")):
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
                jobs.append(self._public_meta(meta))
            jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return jobs

    def _job_dir(self, job_id: str) -> Path:
        if not job_id.replace("-", "").isalnum():
            raise KeyError(job_id)
        path = self.root / job_id
        if not path.exists():
            raise KeyError(job_id)
        return path

    def get(self, job_id: str) -> dict:
        with self._lock:
            meta = json.loads((self._job_dir(job_id) / "job.json").read_text())
            return self._public_meta(meta)

    def progress(self, job_id: str) -> dict:
        with self._lock:
            path = self._job_dir(job_id) / "progress.json"
            if path.exists():
                return json.loads(path.read_text())
            meta = json.loads((self._job_dir(job_id) / "job.json").read_text())
            return {
                "processed_frames": meta.get("frames_processed", 0),
                "total_frames": meta.get("total_frames", 0),
                "percent": 100.0 if meta["status"] == "completed" else 0.0,
                "counted": len(meta.get("results", [])),
            }

    def evidence_png(self, job_id: str, track_id: int, kind: str) -> bytes:
        name = f"piece_{int(track_id)}_{kind}.png"
        path = self._job_dir(job_id) / "evidence" / name
        if not path.exists():
            raise KeyError(name)
        return path.read_bytes()

    def video_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "input.mp4"

    # ------------------------------------------------------------ mutation

    def create(
        self,
        job_id: str,
        filename: str,
        video_bytes: bytes,
        background_bytes: bytes,
        params: VideoParams,
    ) -> dict:
        import datetime
        import uuid

        job_id = job_id or uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / "evidence").mkdir()
        video_path = job_dir / "input.mp4"
        video_path.write_bytes(video_bytes)
        bg_image = imaging.decode_image(background_bytes)
        bg_path = job_dir / "background.png"
        bg_path.write_bytes(_png_bytes(bg_image))

        info = validate_video_file(str(video_path), str(bg_path))
        params.validate(info["width"], info["height"])
        # Fix the calibrated detector snapshot at task start.
        snapshot = self.state.snapshot()

        meta = {
            "job_id": job_id,
            "filename": filename,
            "status": "running",
            "created_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "params": _params_dict(params),
            "total_frames": info["frames"],
            "frames_processed": 0,
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "duration_sec": info["duration_sec"],
            "results": [],
            "tracks": [],
            "merged_frames": [],
            "line_x": params.line_x,
            "roi": list(params.roi),
            "direction": params.direction,
            "detector": {
                "built_at": snapshot.built_at,
                "bank_sha256": snapshot.bank_sha256,
                "threshold": snapshot.threshold,
            },
        }
        self._write_meta(job_id, meta)
        token = CancellationToken()
        with self._lock:
            self.active[job_id] = token
        thread = threading.Thread(
            target=self._run,
            args=(job_id, str(video_path), str(bg_path), params, snapshot, token),
            daemon=True,
        )
        self._thread[job_id] = thread
        thread.start()
        return self._public_meta(meta)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            token = self.active.get(job_id)
            try:
                meta = self.get(job_id)
            except KeyError:
                raise
            if token is None:
                if meta["status"] in ("completed", "error"):
                    raise ValueError(f"任务已结束（{meta['status']}），无法取消")
                # Interrupted in a previous process and recovered on restart.
                if meta["status"] == "interrupted":
                    return meta
                raise ValueError("任务不在运行中")
            token.cancel()
        thread = self._thread.get(job_id)
        if thread is not None:
            thread.join(timeout=30)
        return self.get(job_id)

    # -------------------------------------------------------------- worker

    def _run(
        self,
        job_id: str,
        video_path: str,
        bg_path: str,
        params: VideoParams,
        snapshot,
        token: CancellationToken,
    ) -> None:
        progress = Progress()
        progress.on_update = lambda p: self._write_progress(
            job_id, p.as_dict()
        )
        background_bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
        try:
            output = process_video(
                video_path,
                background_rgb,
                params,
                self.state.extractor,
                snapshot,
                progress,
                token,
            )
            self._finish_success(job_id, output)
        except Exception as exc:
            from .video_pipeline import CancelledError as _Cancelled

            status = "cancelled" if isinstance(exc, _Cancelled) else "error"
            self._finish_failure(job_id, status, exc, progress)
        finally:
            with self._lock:
                self.active.pop(job_id, None)
                self._thread.pop(job_id, None)

    def _finish_success(self, job_id: str, output: dict) -> None:
        evidence_dir = self.root / job_id / "evidence"
        public_results = []
        for record in output["results"]:
            track_id = int(record["track_id"])
            entry = {k: v for k, v in record.items() if k not in ("heat", "crop")}
            entry["has_evidence"] = record.get("crop") is not None
            if record.get("crop") is not None:
                crop_png = _png_bytes(Image.fromarray(record["crop"]))
                heat_png = _png_bytes(Image.fromarray(record["heat"]))
                (evidence_dir / f"piece_{track_id}_crop.png").write_bytes(crop_png)
                (evidence_dir / f"piece_{track_id}_heat.png").write_bytes(heat_png)
            public_results.append(entry)
        with self._lock:
            meta = json.loads((self.root / job_id / "job.json").read_text())
            meta.update(
                {
                    "status": "completed",
                    "frames_processed": output["frames_processed"],
                    "total_frames": output["total_frames"],
                    "results": public_results,
                    "tracks": output["tracks"],
                    "merged_frames": output["merged_frames"],
                    "line_x": output["line_x"],
                    "roi": output["roi"],
                    "direction": output["direction"],
                    "detector": output["detector"],
                }
            )
            self._write_meta(job_id, meta)
            self._write_progress(
                job_id,
                {
                    "processed_frames": output["total_frames"],
                    "total_frames": output["total_frames"],
                    "percent": 100.0,
                    "counted": len(public_results),
                },
            )

    def _finish_failure(self, job_id: str, status: str, exc: Exception, progress) -> None:
        with self._lock:
            path = self.root / job_id / "job.json"
            meta = json.loads(path.read_text())
            meta["status"] = status
            meta["frames_processed"] = progress.processed_frames
            meta["error"] = str(exc) if status == "error" else "用户取消"
            if status == "error":
                meta["traceback"] = traceback.format_exc()[-4000:]
            self._write_meta(job_id, meta)
            self._write_progress(job_id, progress.as_dict())

    # ------------------------------------------------------------ recovery

    def recover_stale(self) -> None:
        """Mark jobs left 'running' by a previous process as interrupted."""
        for meta_path in self.root.glob("*/job.json"):
            meta = json.loads(meta_path.read_text())
            if meta.get("status") == "running":
                meta["status"] = "interrupted"
                meta["error"] = "服务重启导致任务中断，结果不完整"
                self._write_meta(meta["job_id"], meta)

    # -------------------------------------------------------------- helpers

    def _write_meta(self, job_id: str, meta: dict) -> None:
        path = self.root / job_id / "job.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _write_progress(self, job_id: str, payload: dict) -> None:
        path = self.root / job_id / "progress.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    def _public_meta(self, meta: dict) -> dict:
        out = dict(meta)
        out.pop("traceback", None)
        return out


def _params_dict(params: VideoParams) -> dict:
    return {
        "roi": list(params.roi),
        "line_x": params.line_x,
        "direction": params.direction,
        "diff_threshold": params.diff_threshold,
        "min_area": params.min_area,
        "max_piece_area": params.max_piece_area,
        "min_side": params.min_side,
        "max_side": params.max_side,
    }


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
