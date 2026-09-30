"""Development entry point: run the API and web UI on CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "anomaly_inspector.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
