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
