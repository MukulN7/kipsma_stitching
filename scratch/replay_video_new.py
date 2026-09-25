"""Offline video replay diagnostic script with Adaptive Frame Acquisition and Parallel SIFT vs SuperPoint+LightGlue A/B Comparison.

Feeds recorded video frames through the MovementDetector cheap change gate, then passes candidate frames
to two independent registration backends in parallel:
  1. SIFT + RANSAC
  2. SuperPoint + LightGlue

Each backend maintains its own independent last accepted reference image and mosaic state.
Rejection in one backend does not affect the other backend's reference.

Outputs:
  outputs/video_replay_mosaic_sift.tif (.png)
  outputs/video_replay_mosaic_lightglue.tif (.png)

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

from microstitch.movement_detector import MovementDetector
from microstitch.registration import RegistrationEngine, LightGlueRegistrationEngine, RegistrationResult
from microstitch.mosaic import MosaicManager
from microstitch.mosaic_renderer import MosaicCanvasRenderer

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260925_15_18_58_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


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
    print("  PARALLEL ADAPTIVE VIDEO REPLAY STITCHING (SIFT vs SuperPoint+LightGlue)")
    print(f"  Video path: {video_path}")
    print(f"  Resolution: {w}x{h} | Total Frames: {total_video_frames} | FPS: {fps:.2f}")
    print("=" * 95)

    # 1. Cheap movement detector
    movement = MovementDetector(threshold=3.0, auto_update_reference=True)

    # 2. Independent registration backends
    engine_sift = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=150.0,
        use_laplacian=True,
        use_sift=True,
    )
    engine_lightglue = LightGlueRegistrationEngine(
        min_spatial_score=0.35,
        max_shift=150.0,
        max_num_keypoints=1024,
    )

    lg_available = engine_lightglue.is_available()
    if not lg_available:
        print("[WARNING] SuperPoint + LightGlue dependencies unavailable.")
        print("[WARNING] Install with: pip install torch lightglue kornia")
        print("[WARNING] LightGlue backend will mark all registrations as invalid.")

    # 3. Independent mosaic managers & renderers
    mosaic_sift = MosaicManager()
    mosaic_lg = MosaicManager()
    renderer = MosaicCanvasRenderer()

    # Independent reference frames
    prev_accepted_sift: np.ndarray | None = None
    prev_accepted_lg: np.ndarray | None = None

    # Counters and tracking
    source_frame_count = 0
    moved_count = 0

    sift_attempts = 0
    sift_accepted = 0
    sift_rejected = 0

    lg_attempts = 0
    lg_accepted = 0
    lg_rejected = 0

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        source_frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        if source_frame_count % 100 == 0:
            print(f"[PROGRESS] Processed {source_frame_count}/{total_video_frames} source frames...")

        # ── 1. Cheap movement / change gate ──────────────────────────────────
        moved, mad = movement.process_frame(frame)
        if not moved:
            continue

        moved_count += 1

        # ── 2. Independent SIFT Pipeline ─────────────────────────────────────
        if prev_accepted_sift is None:
            mosaic_sift.add_first_frame(frame)
            prev_accepted_sift = frame.copy()
        else:
            sift_attempts += 1
            try:
                reg_sift: RegistrationResult = engine_sift.register(prev_accepted_sift, frame)
            except Exception:
                reg_sift = RegistrationResult(0.0, 0.0, 0.0, False)

            if reg_sift.valid:
                tile = mosaic_sift.add_frame(frame, reg_sift)
                if tile is not None:
                    prev_accepted_sift = frame.copy()
                    sift_accepted += 1
                else:
                    sift_rejected += 1
            else:
                sift_rejected += 1

        # ── 3. Independent SuperPoint + LightGlue Pipeline ──────────────────
        if prev_accepted_lg is None:
            mosaic_lg.add_first_frame(frame)
            prev_accepted_lg = frame.copy()
        else:
            lg_attempts += 1
            try:
                reg_lg: RegistrationResult = engine_lightglue.register(prev_accepted_lg, frame)
            except Exception:
                reg_lg = RegistrationResult(0.0, 0.0, 0.0, False)

            if reg_lg.valid:
                tile = mosaic_lg.add_frame(frame, reg_lg)
                if tile is not None:
                    prev_accepted_lg = frame.copy()
                    lg_accepted += 1
                else:
                    lg_rejected += 1
            else:
                lg_rejected += 1

    cap.release()
    elapsed = time.time() - start_time

    # Render mosaics
    canvas_sift = renderer.render(mosaic_sift, blend=True)
    canvas_lg = renderer.render(mosaic_lg, blend=True)

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_sift = out_dir / "video_replay_mosaic_sift.tif"
    png_sift = out_dir / "video_replay_mosaic_sift.png"
    tif_lg = out_dir / "video_replay_mosaic_lightglue.tif"
    png_lg = out_dir / "video_replay_mosaic_lightglue.png"

    if canvas_sift.size > 0:
        tifffile.imwrite(str(tif_sift), canvas_sift)
        cv2.imwrite(str(png_sift), canvas_sift)
        dims_sift = f"{canvas_sift.shape[1]}x{canvas_sift.shape[0]}"
    else:
        dims_sift = "0x0 (empty)"

    if canvas_lg.size > 0:
        tifffile.imwrite(str(tif_lg), canvas_lg)
        cv2.imwrite(str(png_lg), canvas_lg)
        dims_lg = f"{canvas_lg.shape[1]}x{canvas_lg.shape[0]}"
    else:
        dims_lg = "0x0 (empty)"

    print("\n" + "=" * 95)
    print("  PROCESSING SUMMARY (ADAPTIVE ACQUISITION + PARALLEL A/B)")
    print("=" * 95)
    print(f"  Total Source Frames Evaluated  : {source_frame_count}")
    print(f"  Frames Passing Movement Check  : {moved_count} ({100*moved_count/max(source_frame_count,1):.1f}%)")
    print(f"  Total Processing Time          : {elapsed:.2f} seconds ({source_frame_count/max(elapsed,0.001):.1f} fps)")
    print("-" * 95)
    print("  [BACKEND 1: SIFT + RANSAC]")
    print(f"    Registration Attempts        : {sift_attempts}")
    print(f"    Accepted Registrations       : {sift_accepted}")
    print(f"    Rejected Registrations       : {sift_rejected}")
    print(f"    Final Mosaic Tile Count      : {mosaic_sift.tile_count}")
    print(f"    Final Mosaic Dimensions      : {dims_sift}")
    print("-" * 95)
    print("  [BACKEND 2: SuperPoint + LightGlue]")
    print(f"    Available / Installed        : {lg_available}")
    print(f"    Registration Attempts        : {lg_attempts}")
    print(f"    Accepted Registrations       : {lg_accepted}")
    print(f"    Rejected Registrations       : {lg_rejected}")
    print(f"    Final Mosaic Tile Count      : {mosaic_lg.tile_count}")
    print(f"    Final Mosaic Dimensions      : {dims_lg}")
    print("=" * 95)
    print("\n  OUTPUT FILE PATHS:")
    print(f"  SIFT Mosaic TIFF               : {tif_sift}")
    print(f"  SIFT Mosaic PNG                : {png_sift}")
    print(f"  SuperPoint+LightGlue TIFF      : {tif_lg}")
    print(f"  SuperPoint+LightGlue PNG       : {png_lg}")
    print("=" * 95)


if __name__ == "__main__":
    main()

