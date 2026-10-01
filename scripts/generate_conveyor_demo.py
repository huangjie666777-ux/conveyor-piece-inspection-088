"""Generate a synthetic fixed-camera conveyor video + library images.

Purely procedural (NOT footage of real products). Produces:

- examples/video/conveyor_demo.mp4 (mp4v, 640x320, 20 fps, ~6 s)
- examples/video/background.png (empty belt, same dimensions)
- examples/video/library_normal_*.png (square piece crops for reference/calibration)

The belt moves left to right. Several pieces cross a vertical counting line;
one is scratched/defective. Two pieces touch mid-ROI to exercise merged-
candidate rejection without merging identities or forcing a verdict.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, DURATION = 640, 320, 20, 9.0
OUT_DIR = Path(__file__).resolve().parents[1] / "examples" / "video"
PIECE = 72
ROI = (20, 40, WIDTH - 40, HEIGHT - 80)
LINE_X = 320


def belt_background(seed=7):
    rng = np.random.default_rng(seed)
    yy = np.arange(HEIGHT, dtype=np.float32)[:, None]
    base = 92 + 7 * np.sin(yy * 0.35) + rng.normal(0, 3, (HEIGHT, 1))
    frame = np.repeat(base, WIDTH, axis=1)
    frame = np.stack([frame * 0.85, frame * 0.95, frame], axis=-1)
    cv2.rectangle(frame, (40, 20), (600, 300), (0, 0, 0), 1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def piece_texture(seed, defective=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:PIECE, :PIECE]
    shade = 185 + 10 * np.sin((xx + yy) * 0.12) + rng.normal(0, 4, (PIECE, PIECE))
    rgb = np.stack([shade, shade + 5, shade - 4], axis=-1)
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    cv2.circle(rgb, (PIECE // 2, PIECE // 2), PIECE // 2 - 3, (230, 230, 230), 2)
    if defective:
        cv2.line(rgb, (18, 20), (54, 50), (30, 25, 20), 3)
        cv2.line(rgb, (24, 52), (52, 18), (40, 30, 25), 2)
        cv2.circle(rgb, (50, 48), 6, (25, 20, 15), -1)
    return rgb


def paste_piece(frame, texture, cx, cy):
    x = int(round(cx - PIECE / 2))
    y = int(round(cy - PIECE / 2))
    sx0 = max(0, -x)
    sy0 = max(0, -y)
    dx0 = max(0, x)
    dy0 = max(0, y)
    dx1 = min(WIDTH, x + PIECE)
    dy1 = min(HEIGHT, y + PIECE)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    frame[dy0:dy1, dx0:dx1] = texture[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bg = belt_background()
    cv2.imwrite(str(OUT_DIR / "background.png"), bg)

    # Normal pieces reuse the same random textures used as clean library
    # crops; the single defective piece carries dark scratches/spot.
    normal_textures = [piece_texture(seed) for seed in (301, 302, 303, 304, 305)]
    defective = piece_texture(999, defective=True)

    pieces = [
        (-40, 100, 3.2, normal_textures[0]),
        (120, 225, 3.0, normal_textures[1]),
        (-150, 100, 3.1, defective),
        (-260, 225, 3.3, normal_textures[2]),
        (-420, 100, 3.0, normal_textures[3]),
        (-480, 100, 4.5, normal_textures[4]),  # briefly tailgates #4, clears
    ]
    # Clean library observations, (frame, center_x, center_y): pieces well
    # inside the ROI and not touching any neighbour.
    clean_sources = [
        (44, 101, 64), (30, 210, 189), (165, 75, 64),
        (110, 103, 189), (100, 70, 189), (172, 96, 64),
        (150, 195, 64), (20, 180, 189),
    ]

    total = int(FPS * DURATION)
    video_path = OUT_DIR / "conveyor_demo.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise SystemExit("mp4v encoder unavailable")
    frames = []
    for f in range(total):
        frame = bg.copy()
        for start_x, cy, speed, texture in pieces:
            cx = start_x + speed * f
            if -PIECE <= cx <= WIDTH + PIECE:
                paste_piece(frame, texture, cx, cy)
        cv2.line(frame, (LINE_X, ROI[1]), (LINE_X, ROI[1] + ROI[3]), (0, 200, 0), 2)
        cv2.rectangle(
            frame, (ROI[0], ROI[1]),
            (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0, 200, 255), 1,
        )
        writer.write(frame)
        frames.append(frame)
    writer.release()

    # Library images are cut from the encoded video (or the in-memory frames
    # as a fallback) so codec noise seen at inspection is part of "normal".
    reader = cv2.VideoCapture(str(video_path))
    encoded = {}
    fi = -1
    wanted = {item[0] for item in clean_sources}
    while wanted - set(encoded):
        ok, grabbed = reader.read()
        if not ok:
            break
        fi += 1
        if fi in wanted:
            encoded[fi] = grabbed
    reader.release()

    for i, (frame_idx, cx, cy) in enumerate(clean_sources, start=1):
        source = encoded.get(frame_idx, frames[frame_idx])
        x0 = max(0, int(cx - PIECE / 2) - 6)
        y0 = max(0, int(cy) - 6)
        crop = source[y0:y0 + PIECE + 12, x0:x0 + PIECE + 12]
        cv2.imwrite(str(OUT_DIR / f"library_normal_{i}.png"), crop)

    # Defective truth: crop the encoded defective piece when fully inside ROI.
    defect_frame_idx = 75
    defect_cx = -150 + 3.1 * defect_frame_idx
    defect_source = encoded.get(defect_frame_idx, frames[defect_frame_idx])
    x0 = max(0, int(defect_cx - PIECE / 2) - 6)
    y0 = 64 - 6
    cv2.imwrite(
        str(OUT_DIR / "defective_truth.png"),
        defect_source[y0:y0 + PIECE + 12, x0:x0 + PIECE + 12],
    )
    print(f"wrote demo assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
