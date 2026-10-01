
"""Background video-inspection jobs: progress, cancellation, persistence.

Only one job runs at a time.  The calibrated detector (bank + threshold) is
snapshotted before the worker thread starts and saved inside the job
directory; a later detector rebuild never affects a running or finished task.
Interrupted workers are explicitly marked interrupted on restart; a cancelled
job stops processing and is never reported as completed.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from . import video as video_mod
from .video import MAX_VIDEO_BYTES, VIDEO_EXTENSIONS, VideoConfig, validate_video



class JobError(ValueError):
    pass


class JobManager:
    def __init__(self, jobs_dir, state):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.state = state
        self._lock = threading.RLock()
        self._jobs = {}
        self._threads = {}
        self._events = {}
        self._load_all()

    def create_job(self, filename, video_bytes, background_bytes, config):
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise JobError(f"unsupported video format {ext or 'unknown'}; use MP4 or AVI")
        if len(video_bytes) > MAX_VIDEO_BYTES:
            raise JobError(
                f"video exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MB limit"
            )
        if len(background_bytes) > 10 * 1024 * 1024:
            raise JobError("background image exceeds the 10 MB limit")
        background = cv2.imdecode(
            np.frombuffer(background_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if background is None:
            raise JobError("background image could not be decoded; use PNG/JPEG")

        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()):
                raise JobError("another video job is already running")
            try:
                snapshot = self.state.detector_snapshot()
            except ValueError:
                raise JobError(
                    "detector is not calibrated yet; build the detector first"
                )
            job_id = time.strftime("job-%Y%m%d-%H%M%S-") + str(
                int(time.time() * 1000) % 100000
            )
            job_dir = self.jobs_dir / job_id
            (job_dir / "evidence").mkdir(parents=True)
            video_path = job_dir / f"video{ext}"
            video_path.write_bytes(video_bytes)
            bg_path = job_dir / "background.png"
            cv2.imwrite(str(bg_path), background)

            cap = validate_video(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if background.shape[1] != width or background.shape[0] != height:
                raise JobError(
                    "background image must have the same dimensions as the video"
                )
            video_mod.validate_config(config, width, height)

            torch.save(
                {
                    "version": snapshot.version,
                    "built_at": snapshot.built_at,
                    "bank": snapshot.bank,
                    "threshold": snapshot.threshold,
                },
                job_dir / "detector_snapshot.pt",
            )
            job = {
                "id": job_id,
                "status": "running",
                "filename": filename,
                "video_file": video_path.name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "progress": {"processed": 0, "total": frames},
                "fps": fps,
                "width": width,
                "height": height,
                "config": {
                    "roi": list(config.roi),
                    "line_x": config.line_x,
                    "direction": config.direction,
                },
                "detector": {
                    "version": snapshot.version,
                    "built_at": snapshot.built_at,
                    "threshold": float(snapshot.threshold),
                },
                "pieces": [],
                "tracks": [],
                "error": None,
            }
            self._jobs[job_id] = job
            self._persist(job_id)
            event = threading.Event()
            self._events[job_id] = event
            thread = threading.Thread(
                target=self._run,
                args=(job_id, video_path, bg_path, config, snapshot, event),
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            return self.public_status(job_id)

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] != "running":
                return self.public_status(job_id)
            self._events[job_id].set()
        return self.public_status(job_id)

    def list_jobs(self):
        with self._lock:
            return [
                self.public_status(jid)
                for jid in sorted(self._jobs, reverse=True)
            ]

    def status(self, job_id):
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self.public_status(job_id)

    def paths(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job_dir = self.jobs_dir / job_id
            return {
                "dir": job_dir,
                "video": job_dir / job["video_file"],
                "background": job_dir / "background.png",
            }

    def public_status(self, job_id):
        job = self._jobs[job_id]
        return {
            k: job[k]
            for k in (
                "id", "status", "filename", "created_at", "progress",
                "fps", "width", "height", "config", "detector",
                "pieces", "tracks", "error",
            )
        }

    def _run(self, job_id, video_path, bg_path, config, snapshot, event):
        try:
            background = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)

            def progress(processed, total):
                with self._lock:
                    job = self._jobs[job_id]
                    job["progress"] = {"processed": int(processed), "total": int(total)}
                if processed % 50 < 6:
                    self._persist(job_id)

            output = video_mod.process_video(
                video_path, background, config, self.state,
                snapshot, event, progress,
            )
        except video_mod.CancelledError:
            with self._lock:
                self._jobs[job_id]["status"] = "cancelled"
                self._jobs[job_id]["error"] = "用户取消，处理已停止"
                self._persist(job_id)
            return
        except Exception as exc:
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
                self._jobs[job_id]["traceback"] = traceback.format_exc()
                self._persist(job_id)
            return

        with self._lock:
            job = self._jobs[job_id]
            piece_objects = output.pop("_pieces")
            job_dir = self.jobs_dir / job_id
            for piece in piece_objects:
                if piece.evidence_crop is not None:
                    (job_dir / "evidence" / f"{piece.track_id}_crop.png").write_bytes(
                        piece.evidence_crop
                    )
                if piece.evidence_heatmap is not None:
                    (job_dir / "evidence" / f"{piece.track_id}_heatmap.png").write_bytes(
                        piece.evidence_heatmap
                    )
            job["pieces"] = output["pieces"]
            job["tracks"] = output["tracks"]
            job["progress"] = {"processed": output["frames"], "total": output["frames"]}
            job["status"] = "completed"
            self._persist(job_id)

    def _persist(self, job_id):
        job = self._jobs[job_id]
        payload = {k: v for k, v in job.items() if k != "traceback"}
        tmp = self.jobs_dir / job_id / "job.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.jobs_dir / job_id / "job.json")

    def _load_all(self):
        for job_dir in sorted(self.jobs_dir.iterdir()):
            meta_path = job_dir / "job.json"
            if not job_dir.is_dir() or not meta_path.exists():
                continue
            try:
                job = json.loads(meta_path.read_text())
            except Exception:
                continue
            if job.get("status") == "running":
                job["status"] = "interrupted"
                job["error"] = "服务重启导致任务中断，需重新提交视频"
                meta_path.write_text(json.dumps(job, indent=2))
            self._jobs[job["id"]] = job
