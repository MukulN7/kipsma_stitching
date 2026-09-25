"""Offline video replay diagnostic script with frame-by-frame SIFT registration.

Processes EVERY source frame sequentially:
Frame N -> SIFT registration against the immediately previous source frame (Frame N-1).

If a frame fails registration, skip adding that frame to the mosaic,
but continue comparing the next source frame (Frame N+1) against Frame N.

Outputs:
  outputs/video_replay_mosaic_sift.tif (.png)

Usage:
    python scratch/replay_video.py [video_path]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.registration import RegistrationEngine, RegistrationResult
from microstitch.mosaic import MosaicManager
from microstitch.mosaic_renderer import MosaicCanvasRenderer

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260925_15_21_07_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


def _save_canvas(tif_path: Path, png_path: Path, canvas: np.ndarray) -> str:
    """Robustly normalizes and writes mosaic canvas to TIFF and PNG formats."""
    if canvas is None or canvas.size == 0:
        return "0x0 (empty)"

    # Ensure C-contiguous array
    arr = np.ascontiguousarray(canvas)

    # Normalize dtype to uint8 if needed
    if arr.dtype != np.uint8:
        mn, mx = float(arr.min()), float(arr.max())
        rng = mx - mn if mx != mn else 1.0
        arr = np.clip((arr.astype(np.float32) - mn) / rng * 255.0, 0, 255).astype(np.uint8)

    str_tif = str(tif_path.resolve())
    str_png = str(png_path.resolve())

    # Try cv2.imwrite for TIFF first (C++ native writer), fallback to tifffile with RGB conversion
    written_tif = cv2.imwrite(str_tif, arr)
    if not written_tif:
        try:
            if arr.ndim == 3 and arr.shape[2] == 3:
                rgb_arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                rgb_arr = arr
            tifffile.imwrite(str_tif, rgb_arr)
            written_tif = True
        except Exception:
            pass

    # Save PNG preview
    cv2.imwrite(str_png, arr)

    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    return f"{w}x{h} ({ch} channels)"


def main() -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 95)
    print("  FRAME-BY-FRAME SEQUENTIAL VIDEO REPLAY STITCHING (SIFT)")
    print(f"  Video path: {video_path}")
    print(f"  Resolution: {w}x{h} | Total Frames: {total_video_frames} | FPS: {fps:.2f}")
    print("=" * 95)

    # Registration Engine (SIFT)
    engine_sift = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=150.0,
        use_laplacian=True,
        use_sift=True,
    )

    # Mosaic Manager & Renderer
    mosaic_sift = MosaicManager()
    renderer = MosaicCanvasRenderer()

    prev_source_frame: np.ndarray | None = None

    source_frame_count = 0
    sift_attempts = 0
    sift_accepted = 0
    sift_rejected = 0

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        source_frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        if source_frame_count % 200 == 0:
            print(f"[PROGRESS] Processed {source_frame_count}/{total_video_frames} source frames...")

        if prev_source_frame is None:
            mosaic_sift.add_first_frame(frame)
            prev_source_frame = frame.copy()
        else:
            sift_attempts += 1
            try:
                reg_sift: RegistrationResult = engine_sift.register(prev_source_frame, frame)
            except Exception:
                reg_sift = RegistrationResult(0.0, 0.0, 0.0, False)

            if reg_sift.valid:
                tile = mosaic_sift.add_frame(frame, reg_sift)
                if tile is not None:
                    sift_accepted += 1
                else:
                    sift_rejected += 1
            else:
                sift_rejected += 1

            # Update reference to immediately previous source frame
            prev_source_frame = frame.copy()

    cap.release()
    elapsed = time.time() - start_time
    processing_fps = source_frame_count / max(elapsed, 0.001)

    # Render mosaic
    canvas_sift = renderer.render(mosaic_sift, blend=True)

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_sift = out_dir / "video_replay_mosaic_sift.tif"
    png_sift = out_dir / "video_replay_mosaic_sift.png"

    dims_sift = _save_canvas(tif_sift, png_sift, canvas_sift)

    print("\n" + "=" * 95)
    print("  PROCESSING SUMMARY")
    print("=" * 95)
    print(f"  Total Frames                 : {source_frame_count}")
    print(f"  SIFT Attempts                : {sift_attempts}")
    print(f"  Accepted                     : {sift_accepted}")
    print(f"  Rejected                     : {sift_rejected}")
    print(f"  Final Mosaic Dimensions      : {dims_sift}")
    print(f"  Processing FPS               : {processing_fps:.2f}")
    print("=" * 95)
    print("\n  OUTPUT FILE PATHS:")
    print(f"  SIFT Mosaic TIFF             : {tif_sift}")
    print(f"  SIFT Mosaic PNG              : {png_sift}")
    print("=" * 95)


if __name__ == "__main__":
    main()
