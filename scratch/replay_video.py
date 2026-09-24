"""Offline video replay diagnostic script.

Feeds a recorded video file through the MicroStitch stitching pipeline
(MovementDetector -> RegistrationEngine -> MosaicManager -> MosaicCanvasRenderer)
to evaluate baseline performance on actual microscope motion recordings.

Usage:
    python scratch/replay_video.py [video_path]
Default video_path:
    C:\\Users\\mukul\\OneDrive\\Pictures\\Camera Roll 1\\WIN_20260922_18_07_08_Pro.mp4
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
from microstitch.registration import RegistrationEngine, RegistrationResult
from microstitch.mosaic import MosaicManager
from microstitch.mosaic_renderer import MosaicCanvasRenderer

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260922_18_07_08_Pro.mp4"
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
    print(f"  OFFLINE VIDEO REPLAY STITCHING BASELINE")
    print(f"  Video path: {video_path}")
    print(f"  Resolution: {w}x{h} | Total Frames: {total_video_frames} | FPS: {fps:.2f}")
    print("=" * 95)

    movement = MovementDetector(threshold=3.0, auto_update_reference=True)
    engine   = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=150.0,
        use_laplacian=True,
    )
    mosaic   = MosaicManager()
    renderer = MosaicCanvasRenderer()

    source_frame_count = 0
    prev_accepted: np.ndarray | None = None
    processed_count = 0
    moved_count = 0
    registration_attempts = 0
    accepted_count = 0
    rejected_count = 0

    reg_log: list[dict] = []

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        source_frame_count += 1
        if (source_frame_count - 1) % 5 != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        processed_count += 1

        if processed_count % 50 == 0:
            print(f"[PROGRESS] Processed {processed_count} frames...")

        moved, mad = movement.process_frame(frame)

        if not moved:
            continue

        moved_count += 1

        # First moved frame placed as origin
        if prev_accepted is None:
            mosaic.add_first_frame(frame)
            prev_accepted = frame.copy()
            reg_log.append({
                "frame": processed_count,
                "mad": mad,
                "dx": 0.0,
                "dy": 0.0,
                "response": 1.0,
                "spatial": 1.0,
                "status": "ORIGIN",
                "valid": True,
            })
            continue

        registration_attempts += 1

        try:
            reg: RegistrationResult = engine.register(prev_accepted, frame)
        except Exception as exc:
            rejected_count += 1
            reg_log.append({
                "frame": processed_count,
                "mad": mad,
                "dx": 0.0,
                "dy": 0.0,
                "response": 0.0,
                "spatial": 0.0,
                "status": f"ERROR: {exc}",
                "valid": False,
            })
            continue

        if reg.valid:
            tile = mosaic.add_frame(frame, reg)
            if tile is not None:
                prev_accepted = frame.copy()
                accepted_count += 1
                status = "ACCEPTED"
            else:
                rejected_count += 1
                status = "REJECTED (MosaicManager)"
        else:
            rejected_count += 1
            reasons = []
            if reg.response < engine.min_response:
                reasons.append(f"resp={reg.response:.4f}<{engine.min_response}")
            if reg.spatial_score < engine.min_spatial_score:
                reasons.append(f"spatial={reg.spatial_score:.4f}<{engine.min_spatial_score}")
            status = "REJECTED (" + ", ".join(reasons) + ")"

        reg_log.append({
            "frame": processed_count,
            "mad": mad,
            "dx": reg.dx,
            "dy": reg.dy,
            "response": reg.response,
            "spatial": reg.spatial_score,
            "status": status,
            "valid": reg.valid,
        })

    cap.release()
    elapsed = time.time() - start_time

    # Render final mosaic with feather blending
    final_canvas = renderer.render(mosaic, blend=True)

    # Prepare outputs
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_path = out_dir / "video_replay_mosaic.tif"
    png_path = out_dir / "video_replay_mosaic.png"

    if final_canvas.size > 0:
        tifffile.imwrite(str(tif_path), final_canvas)
        cv2.imwrite(str(png_path), final_canvas)
        canvas_dims = f"{final_canvas.shape[1]}x{final_canvas.shape[0]} ({final_canvas.ndim} channels)"
    else:
        canvas_dims = "0x0 (empty)"

    # Print summary table
    print("\n" + "=" * 95)
    print("  PROCESSING SUMMARY")
    print("=" * 95)
    print(f"  Total source frames      : {source_frame_count}")
    print(f"  Processed frame count    : {processed_count}")
    print(f"  Frames with movement     : {moved_count} ({100*moved_count/max(processed_count,1):.1f}%)")
    print(f"  Frames reaching engine   : {registration_attempts}")
    print(f"  Accepted registrations   : {accepted_count}")
    print(f"  Rejected registrations   : {rejected_count}")
    print(f"  Final mosaic tile count  : {mosaic.tile_count}")
    print(f"  Final mosaic dimensions  : {canvas_dims}")
    print(f"  Total processing time    : {elapsed:.2f} seconds ({processed_count/max(elapsed,0.001):.1f} fps)")

    # Print first 20 registration results
    print("\n" + "=" * 95)
    print(f"  FIRST {min(20, len(reg_log))} REGISTRATION RESULTS")
    print("=" * 95)
    print(f"{'Frame':>6} | {'MAD':>6} | {'dx':>8} | {'dy':>8} | {'resp':>7} | {'spatial':>7} | {'valid':>6} | status")
    print("-" * 95)

    for entry in reg_log[:20]:
        v_str = "OK" if entry["valid"] else "REJECT"
        print(f"{entry['frame']:>6} | {entry['mad']:>6.2f} | {entry['dx']:>8.2f} | {entry['dy']:>8.2f} | "
              f"{entry['response']:>7.4f} | {entry['spatial']:>7.4f} | {v_str:>6} | {entry['status']}")

    print("=" * 95)
    print("\n  OUTPUT FILE PATHS:")
    print(f"  TIFF Mosaic: {tif_path}")
    print(f"  PNG Preview: {png_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
