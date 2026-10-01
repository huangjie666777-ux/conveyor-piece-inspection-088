# Industrial anomaly inspection

FastAPI + frozen ResNet18 surface anomaly inspection, plus conveyor video
per-piece tracking, counting and evidence review. CPU-only and offline.

Python 3.10.12 is available in WSL. Use `.venv/bin/python` for this project. PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, FastAPI 0.115.12 and other exact dependencies are installed in the local `.venv`; see requirements.lock. Browser code may use native JavaScript without a Node build toolchain.

The pretrained ResNet18 IMAGENET1K_V1 weights are already present at `models/resnet18-f37072fd.pth`. Source and checksum are in models/manifest.json. No cloud API, credentials or GPU are required. The binary is omitted from Git and included in the prepared local directory.

To recreate the dependency environment after a fresh Git clone:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth -o models/resnet18-f37072fd.pth
```

Verify the downloaded file against the SHA256 in models/manifest.json. Package and weight downloads require network access; application inference can use the prepared local assets offline. The intended image preprocessing differs from the classification weight's default center crop: follow the requested whole-image preprocessing when implementing the application.

## Running the application

```sh
.venv/bin/python scripts/generate_examples.py   # one-time, writes examples/
.venv/bin/python scripts/run_server.py          # http://127.0.0.1:8000
```

Workflow in the page (or via the API under `/api`):

1. Upload reference normal PNG/JPEG images and independent calibration normal PNG/JPEG images. Each group has previews and per-sample delete buttons; identical image content is rejected both within and across groups. Limit: PNG/JPEG only, 10 MB per file.
2. Click **重建检测器**. Every image is resized whole to 224x224 (bilinear, no center crop) and ImageNet-normalized. A frozen ResNet18 provides layer2 (128-d) and layer3 (256-d) features; layer3 is bilinearly aligned to the layer2 28x28 grid and concatenated into 384-d local descriptors.
3. Reference descriptors are subsampled to at most 256 items with deterministic greedy farthest-point selection (streaming minimum distances; no full pairwise matrix). The threshold is the 95% linear-interpolated quantile of image scores (max local nearest-neighbour Euclidean distance) computed on the calibration images only.
4. Inspect a new image: the score, threshold, verdict and an original-image heatmap overlay with an adjustable alpha are shown. The low-resolution distance map is bilinearly upscaled to the original size and colorized with a fixed upper bound of `2 x threshold`; a threshold of zero is reported explicitly and falls back to the image maximum. State (samples, memory bank, threshold, preprocessing metadata) persists under `data/`; failed rebuilds keep the previous bank, and each detection snapshots one bank version under a lock.

`examples/` contains actionable synthetic images (procedural brushed-metal-like textures, not real products): normal images for the two normal sets, plus scratch and foreign-object images for detection demos only.

Run tests with `.venv/bin/python -m pytest -q`.

## Conveyor video piece tracking (video quality inspection)

The application also tracks individual work pieces crossing a fixed camera's conveyor belt, keeping the original single-image inspection flow unchanged. All processing runs offline on CPU and frames are streamed one at a time (OpenCV `VideoCapture`); the video is never loaded into memory as a whole.

### Generate the synthetic demo

```sh
.venv/bin/python scripts/generate_video_demo.py
```

This writes under `examples/video/`:

- `conveyor_demo.mp4` — 480x270, 10 fps, left-to-right belt with four independent metal-like pieces (one defective) plus one pair of pieces that touch and therefore form a merged candidate blob;
- `conveyor_background.png` — the matching empty-belt background (same dimensions, required);
- `bank/reference_*.png`, `bank/calibration_*.png` — normal isolated-piece crops (taken from real empty-belt frames) for building the detector; `bank/defective_demo_do_not_calibrate.png` is for detection demos only;
- `params_hint.json` — ROI / counting-line / size parameters matched to the demo.

### Supported uploads and validation

- Video: **MP4 / MOV / AVI**, up to **200 MB**, **5–30 fps**, short side at least 240 px, long side at most 1920 px, duration at most 120 s. The background must be a **PNG/JPEG of exactly the video frame size**. Validation (`POST /api/video/validate`) decodes only metadata plus one frame and returns suggested ROI/line/size parameters.
- The ROI rectangle, vertical counting line x coordinate and travel direction (`ltr` / `rtl`) are validated against the frame size; the counting line must lie inside the ROI. Piece area/side/aspect parameters are validated too. The endpoint rejects wrong formats, oversize files, undecodable videos, missing frame-count metadata and size mismatches with explicit Chinese error messages.

### How tracking and counting work

- **Segmentation**: grayscale Gaussian blur + absolute background difference, binary threshold, open/close morphology to remove noise specks, then connected components inside the ROI. Blobs failing the area/side/aspect envelope are discarded; oversized blobs are explicitly marked **merged** (touching pieces / intrusion), never split heuristically, never fed to the tracker and never auto-passed.
- **Tracking**: each target uses a constant-velocity Kalman filter on centroid/scale with a global one-to-one Hungarian assignment (`scipy.optimize.linear_sum_assignment`). A match must pass both a squared-Mahalanobis motion gate and a cosine gate on appearance descriptors (mean pooled frozen ResNet layer2+layer3 features). Gated-out pairs are left unmatched instead of being forced. Tracks survive short missed detections by coasting on the Kalman prediction; tracks that miss for too long are closed.
- **Counting**: a piece is counted only when two **real observations** (never a predicted box) straddle the vertical line in the configured direction. A confirmed track is counted at most once, so jitter and re-crossing the line cannot double-count; reverse-direction motion is ignored.
- **Evidence**: when a track crosses, its most recent complete, non-merged in-frame crop (with margin) is sent through the **same original anomaly algorithm** (memory bank + calibrated threshold + heatmap). Each result stores track ID, video time, frame, score, verdict (`ok` / `anomaly` / `review_required`), the original crop and the evidence heat map PNG. If no clean crop exists (e.g. the piece only ever appeared in a merged blob) or inference fails, the piece is marked **待复核 (review required)** — it is never silently passed.
- **Fixed detector per task**: a job snapshots the calibrated detector (memory-bank tensor, threshold, build time and bank content hash) at start; rebuilding the bank afterwards never changes the detector version used by that job. The job JSON records the bank hash it ran with.

### Atomic detector builds

Memory banks are stored as content-addressed files under `data/banks/<sha256>.pt`. A rebuild writes the new bank file first, then publishes threshold + bank file name/hash with a single atomic metadata rename (fsync + `os.replace`). A failed rebuild keeps the previous bank; after a crash/restart, metadata referencing a missing or hash-mismatching bank is loaded as "not built" instead of pairing a stale threshold with an unrelated bank (this fixes the earlier two-file commit/restart mixing bug). Old unreferenced bank files are pruned on startup.

### Background jobs, progress, cancellation and recovery

- `POST /api/jobs` starts a background worker thread; `GET /api/jobs/{id}/progress` polls frame progress and counted pieces; `POST /api/jobs/{id}/cancel` sets a cooperative token checked per frame — the worker stops decoding and the job is marked `cancelled` (never `completed`).
- Jobs persist under `data/jobs/` (uploaded video, background, `job.json`, evidence PNGs, periodic `progress.json`). Completed jobs can be replayed after a server restart. Jobs left `running` by a dead process are marked **中断 (`interrupted`)** on startup and cannot masquerade as finished results.
- Replay (`GET /` page, section ⑤) serves the original video (HTTP range support for seeking) and overlays the ROI (dashed), counting line/direction arrow, per-track coloured real-observation trajectories with IDs, and a red warning on merged-blob frames. Clicking a piece chip seeks the video to its crossing time and shows its score, threshold, verdict, crop and heat map; review-required pieces show the reason instead of evidence.

### API summary

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/video/limits` | supported formats/size/fps/duration |
| POST | `/api/video/validate` | validate video+background, return metadata and suggested parameters |
| POST | `/api/jobs` | start a video job (multipart files + ROI/line/direction/size params) |
| GET | `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/progress` | list / detail / progress |
| POST | `/api/jobs/{id}/cancel` | cancel a running job |
| GET | `/api/jobs/{id}/video` | original video with byte-range support |
| GET | `/api/jobs/{id}/evidence/{track_id}/{crop|heat}.png` | piece evidence images |

### Known limitations

- Background subtraction assumes a truly static camera and an empty-belt background frame under similar lighting; strong illumination change, shadows or camera motion create false blobs. There is no online background adaptation.
- Touching/overlapping pieces are **not split**: the merged blob is excluded from tracking and from automatic verdicts by design; inspectors must handle those pieces separately (they appear as merged-frame warnings / uncounted pieces, and any track whose only evidence is merged is review-required).
- Evidence uses a single 2D crop; heavily occluded, border-touching or non-upright parts receive weaker crops. The pipeline is a QA assist, not a safety-certified measurement system.
- CPU only; the 200 MB / 120 s limits keep processing time bounded. Very dense scenes increase ResNet crop-feature cost linearly with the number of clean blobs.
- Only MP4/MOV/AVI containers decodable by the installed OpenCV build are supported (the prepared environment encodes/decodes `mp4v`); variable-FPS content is indexed by frame index / reported average FPS.

## Additional local vision dependencies

The project virtual environment also contains opencv-python-headless 4.11.0.86 and SciPy 1.15.3, alongside the existing CPU PyTorch encoder. Invoke `.venv/bin/python` directly. These packages provide video decoding and numerical operations; no video tracking functionality is preimplemented. Exact installed dependencies are pinned in `requirements.lock`. To recreate the environment, install that lockfile with the PyTorch CPU package index and obtain the local encoder weights as described above.
