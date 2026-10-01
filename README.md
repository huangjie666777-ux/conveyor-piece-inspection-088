# Industrial anomaly inspection

Dependency scaffold only. No application or anomaly detection implementation exists yet.

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

## Additional local vision dependencies

The project virtual environment also contains opencv-python-headless 4.11.0.86 and SciPy 1.15.3, alongside the existing CPU PyTorch encoder. Invoke `.venv/bin/python` directly. These packages provide video decoding and numerical operations; no video tracking functionality is preimplemented. Exact installed dependencies are pinned in `requirements.lock`. To recreate the environment, install that lockfile with the PyTorch CPU package index and obtain the local encoder weights as described above.

## Conveyor video per-piece tracking QC

In addition to single-image inspection, section ⑤ of the page runs fixed-camera conveyor video quality control:

1. Upload a video (**MP4/AVI, ≤ 200 MB, 1–120 fps, frame sides 120–3840 px**) and an empty-belt background image (**PNG/JPEG, ≤ 10 MB, exactly the video dimensions**). Limits are also returned by `GET /api/video/limits`.
2. Drag a rectangle for the detection ROI and click to place the vertical counting line; choose left-to-right (`lr`) or right-to-left (`rl`). Submit starts a **background job** (`POST /api/jobs`); progress is polled from `GET /api/jobs/{id}` and `POST /api/jobs/{id}/cancel` stops processing (a cancelled job is explicitly `cancelled`, never reported as completed).
3. Decoding is strictly one frame at a time (`cv2.VideoCapture.read`); the video is never loaded into memory. Per frame: grayscale blur → absdiff against the empty background → threshold + open/close morphology → contour filtering by area, side length and fill ratio. Oversized blobs and blobs that erosion splits into two cores are flagged **merged**, excluded from association, and never force a verdict.
4. Identity uses a constant-velocity `cv2.KalmanFilter` on `(cx, cy, w, h)`, the existing ResNet18 feature extractor pooled over each padded crop as an L2 appearance descriptor, and SciPy Hungarian one-to-one assignment on Mahalanobis + appearance cost with hard gates. Missed detections coast up to 8 frames; predicted boxes are drawn dashed and can **never** trigger counting. A piece counts once, on a real observation, only when its previous real centre crosses the line in the configured direction (jitter/re-crossing is latched off; tracks first seen beyond the line cannot count).
5. At crossing, the most recent **complete, non-merged, ROI-interior, line-clear** crop is checked with the original anomaly algorithm; track ID, video time, score, fixed-snapshot threshold, verdict (`ok` / `defect` / `review`) and crop + heatmap evidence are stored. Missing/unreliable evidence becomes `review` (待复核) rather than a pass/fail guess.

### Detector snapshot and atomic commit

When a job starts, the active bank + threshold are snapshotted once (`AppState.detector_snapshot`, also saved under `data/jobs/<id>/detector_snapshot.pt`). Rebuilding the detector later never changes a running or finished task. The bank and threshold are now committed together as one atomic `data/detector.pt` bundle (temp file + fsync + single rename), fixing the previous failure/restart window where independent `memory_bank.pt` and `state.json` renames could pair a new bank with an old threshold; legacy files are still read on startup.

### Results, restart, playback

Completed jobs persist under `data/jobs/<id>/` (`job.json`, the uploaded video, background, evidence PNGs, detector snapshot) and remain viewable after restart; jobs found `running` on startup are explicitly marked `interrupted`. The page plays the uploaded video with a synchronized canvas overlay (ROI, counting line, observed vs coasted track boxes, IDs); clicking a result row seeks to the crossing instant and opens the stored crop and anomaly heatmap. Evidence is served from `/api/jobs/{id}/evidence/{track}/{crop|heatmap}` and the video supports HTTP range requests.

### Synthetic demo assets

`.venv/bin/python scripts/generate_conveyor_demo.py` writes `examples/video/`:

- `conveyor_demo.mp4` — 640×320 @ 20 fps, five pieces cross left-to-right (four normal, one scratched/defective); two pieces briefly tailgate/merge inside the ROI;
- `background.png` — matching empty-belt frame;
- `library_normal_1..8.png` — clean codec-consistent piece crops cut from the encoded video; use 1–4 as reference, 5–8 as calibration;
- `defective_truth.png` — the scratched piece crop (verification only, never calibrate on it).

ROI for the demo: x=20 y=40 w=600 h=240, counting line x=320, direction left-to-right.

### Known limitations

- Research-grade background subtraction: assumes a fixed camera, a representative empty-belt image, stable lighting and a flat belt; shadows, reflections and heavy compression can over/under-segment. Thresholds are fixed defaults tuned for the demo; real deployments need per-line tuning.
- Merged/touching candidates are withheld rather than split, so a piece only seen while merged can miss a clean verdict and is marked `review`; there is no instance segmentation.
- Appearance is a single mean pooled ResNet descriptor per crop; identical-looking parts rely mostly on motion gating, and long full occlusions beyond the coast window start new identities.
- Evidence crops and tracks are tied to the configured ROI/line; a bad line placement or ROI clipping biases counts.
- Single background worker (one running job at a time); CPU-only throughput depends on piece count per frame (one ResNet pass per clean detection per frame).
