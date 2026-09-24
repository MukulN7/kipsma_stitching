"""Manual movement diagnostic script for USB camera registration.

Continuously captures frames from USB camera index 1 at 640x360,
registers current frame against reference frame using RegistrationEngine,
and displays/logs live metrics (dx, dy, magnitude, response, spatial score).

Keys:
  R - reset reference frame
  Q / ESC - quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.frame_source import USBCameraSource
from microstitch.registration import RegistrationEngine, RegistrationResult

cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
width   = int(sys.argv[2]) if len(sys.argv) > 2 else 640
height  = int(sys.argv[3]) if len(sys.argv) > 3 else 360


def run_manual_movement_diagnostic() -> None:
    camera = USBCameraSource(cam_idx, width=width, height=height)
    if not camera.is_opened():
        print(f"[ERROR] Cannot open camera {cam_idx}", file=sys.stderr)
        sys.exit(1)

    engine = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        use_laplacian=True,
    )

    print("=" * 95)
    print(f"  MANUAL MOVEMENT DIAGNOSTIC  cam={cam_idx}  res={width}x{height}")
    print("  Move microscope slide manually to observe dx/dy displacement.")
    print("  Keys: R = Reset reference frame  |  Q = Quit")
    print("=" * 95)
    print(f"{'Frame':>6} | {'Time(s)':>7} | {'dx':>8} | {'dy':>8} | {'mag(px)':>8} | {'resp':>7} | {'spatial':>7} | {'valid':>6} | status")
    print("-" * 95)

    ref_frame: np.ndarray | None = None
    start_time = time.time()
    frame_count = 0

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        elapsed = time.time() - start_time

        if ref_frame is None:
            ref_frame = frame.copy()
            print(f"{frame_count:>6} | {elapsed:>7.2f} | {'---':>8} | {'---':>8} | {'---':>8} | {'---':>7} | {'---':>7} | {'REF':>6} | REFERENCE FRAME SET")

        reg: RegistrationResult = engine.register(ref_frame, frame)
        mag = float(np.hypot(reg.dx, reg.dy))
        valid_str = "OK" if reg.valid else "REJECT"

        status_str = f"dx={reg.dx:+.2f} dy={reg.dy:+.2f} mag={mag:.2f} resp={reg.response:.3f} spat={reg.spatial_score:.3f}"
        print(f"{frame_count:>6} | {elapsed:>7.2f} | {reg.dx:>8.2f} | {reg.dy:>8.2f} | {mag:>8.2f} | {reg.response:>7.4f} | {reg.spatial_score:>7.4f} | {valid_str:>6} | {status_str}")
        sys.stdout.flush()

        # Live OpenCV window display
        disp = frame.copy()
        if disp.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        lines = [
            f"Frame: {frame_count} | Time: {elapsed:.1f}s",
            f"dx: {reg.dx:+.2f} px  dy: {reg.dy:+.2f} px",
            f"mag: {mag:.2f} px  [{valid_str}]",
            f"resp: {reg.response:.3f}  spatial: {reg.spatial_score:.3f}",
            "R=Reset ref  Q=Quit",
        ]

        y = 25
        for line in lines:
            cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Manual Movement Diagnostic", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\n[INFO] User requested quit.")
            break
        elif key in (ord('r'), ord('R')):
            ref_frame = frame.copy()
            print(f"\n[INFO] Reference frame reset at frame {frame_count}.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_manual_movement_diagnostic()
