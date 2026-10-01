"""Generate a synthetic conveyor video plus piece images for the bank.

Procedural content (NOT real products):
- conveyor_background.png: empty fixed-camera belt frame;
- conveyor_demo.mp4: metal-like pieces moving left-to-right, including one
  defective piece and one frame range where two pieces touch (a merged blob);
- bank/: normal isolated-piece crops for reference/calibration uploads.

Recommended job parameters are written to params_hint.json so the merged
pair is rejected as a cluster while single pieces pass the size envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 480, 270
FPS = 10
FRAMES = 100
PIECE = 52
SPEED = 8.0


def belt(seed: int = 7, h: int = HEIGHT, w: int = WIDTH) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.arange(h, dtype=np.float32)[:, None]
    base = 72.0 + 6.0 * np.sin(yy * 0.35)
    frame = np.repeat(base, w, axis=1)
    frame += rng.normal(0, 3.0, size=(h, w))
    rgb = np.stack(
        [frame * 0.95, frame, frame * 1.04], axis=-1
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def piece_texture(seed: int, size: int = PIECE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.arange(size, dtype=np.float32)[:, None]
    xx = np.arange(size, dtype=np.float32)[None, :]
    gradient = 14.0 * (xx / size) + 8.0 * (yy / size)
    stripes = 8.0 * np.sin(yy * 0.6 + rng.uniform(0, 6.28))
    noise = rng.normal(0, 3.5, size=(size, size))
    value = 172.0 + gradient + stripes + noise
    rgb = np.stack([value - 6, value, value + 8], axis=-1)
    tex = np.clip(rgb, 0, 255).astype(np.uint8)
    # Rounded mask so blobs read as compact parts, not rectangles.
    mask = np.zeros((size, size), np.uint8)
    cv2.rectangle(mask, (3, 3), (size - 4, size - 4), 255, -1)
    cv2.circle(mask, (3, 3), 4, 255, -1)
    cv2.circle(mask, (size - 4, 3), 4, 255, -1)
    cv2.circle(mask, (3, size - 4), 4, 255, -1)
    cv2.circle(mask, (size - 4, size - 4), 4, 255, -1)
    return tex, mask


def add_defect(tex: np.ndarray, seed: int = 9001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = tex.copy()
    x0, y0 = 8, int(rng.integers(8, 18))
    x1, y1 = 44, int(rng.integers(34, 46))
    for t in np.linspace(0, 1, 200):
        x = int(x0 + (x1 - x0) * t)
        y = int(y0 + (y1 - y0) * t + 2 * np.sin(t * 10))
        for dy in (-1, 0, 1):
            yy = y + dy
            if 0 <= yy < out.shape[0] and 0 <= x < out.shape[1]:
                out[yy, x] = (out[yy, x] * 0.3 + 15).astype(np.uint8)
    out[10:18, 30:40] = (out[10:18, 30:40] * 0.25 + 20).astype(np.uint8)
    return out


def paste_piece(frame: np.ndarray, tex: np.ndarray, mask: np.ndarray, cx: int, cy: int) -> None:
    h, w = frame.shape[:2]
    half = PIECE // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + PIECE, y1 + PIECE
    for yy in range(max(0, y1), min(h, y2)):
        for xx in range(max(0, x1), min(w, x2)):
            if mask[yy - y1, xx - x1]:
                frame[yy, xx] = tex[yy - y1, xx - x1]


def isolated_piece_image(seed: int, defective: bool = False, canvas: int = 68) -> np.ndarray:
    base = belt(1000 + (seed % 7), canvas, canvas)
    tex, mask = piece_texture(seed)
    if defective:
        tex = add_defect(tex, 9000 + seed)
    rng = np.random.default_rng(seed * 31 + 5)
    jitter_x = int(rng.integers(-3, 4))
    jitter_y = int(rng.integers(-3, 4))
    off = (canvas - PIECE) // 2
    off_x = off + jitter_x
    off_y = off + jitter_y
    region = base[off_y : off_y + PIECE, off_x : off_x + PIECE]
    region[mask > 0] = tex[mask > 0]
    return base


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "examples" / "video"
    (out / "bank").mkdir(parents=True, exist_ok=True)
    background = belt()
    Image.fromarray(background).save(out / "conveyor_background.png")

    # Timeline: (label, start_frame, y, seed, defective, touching)
    timeline = [
        ("A", -2, 80, 11, False, False),
        ("B", 8, 170, 22, True, False),
        ("C", 26, 120, 33, False, True),
        ("D", 26, 120 + PIECE - 4, 44, False, True),
        ("E", 56, 210, 55, False, False),
    ]
    writer = cv2.VideoWriter(
        str(out / "conveyor_demo.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    for f in range(FRAMES):
        frame = background.copy()
        for label, start, y, seed, defective, _touch in timeline:
            cx = int(-PIECE + (f - start) * SPEED)
            if cx < -PIECE or cx > WIDTH + PIECE:
                continue
            tex, mask = piece_texture(seed)
            if defective:
                tex = add_defect(tex)
            paste_piece(frame, tex, mask, cx, y)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Bank pieces: real empty-belt patches taken from the finished video
    # (frames before pieces enter / after they leave) with a clean piece
    # pasted at the centre plus small jitter. This matches the crop margin
    # and background statistics of the tracker's evidence crops.
    capture = cv2.VideoCapture(str(out / "conveyor_demo.mp4"))
    empty_frames = []
    for idx in (0, 1, 2, 97, 98, 99):
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = capture.read()
        if ok:
            empty_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    def bank_crop(seed: int, defective: bool = False) -> np.ndarray:
        rng = np.random.default_rng(seed)
        source = empty_frames[seed % len(empty_frames)]
        cx = int(rng.integers(120, 360))
        cy = int(rng.integers(40, 220))
        canvas = source[cy - 34 : cy + 34, cx - 34 : cx + 34].copy()
        tex, mask = piece_texture(seed)
        if defective:
            tex = add_defect(tex, 9000 + seed)
        off = 8 + int(rng.integers(-3, 4))
        region = canvas[off : off + PIECE, off : off + PIECE]
        region[mask > 0] = tex[mask > 0]
        return canvas

    ref_seeds = list(range(101, 117))
    cal_seeds = list(range(201, 213))
    for i, seed in enumerate(ref_seeds, 1):
        Image.fromarray(bank_crop(seed)).save(
            out / "bank" / f"reference_{i}.png"
        )
    for i, seed in enumerate(cal_seeds, 1):
        Image.fromarray(bank_crop(seed)).save(
            out / "bank" / f"calibration_{i}.png"
        )
    Image.fromarray(bank_crop(222, defective=True)).save(
        out / "bank" / "defective_demo_do_not_calibrate.png"
    )

    hint = {
        "video": "conveyor_demo.mp4",
        "background": "conveyor_background.png",
        "roi": [30, 20, 450, 250],
        "line_x": 240,
        "direction": "ltr",
        "diff_threshold": 40,
        "min_area": 900,
        "max_piece_area": 4200,
        "min_side": 30,
        "max_side": 90,
        "note": "max_side 90 keeps single 52px parts but rejects the touching pair blob",
        "expected": {
            "individual_pieces": 4,
            "defective_track": "B",
            "touching_pair": ["C", "D"],
        },
    }
    (out / "params_hint.json").write_text(json.dumps(hint, indent=2, ensure_ascii=False))
    print(f"wrote synthetic conveyor demo to {out}")


if __name__ == "__main__":
    main()
