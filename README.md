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
