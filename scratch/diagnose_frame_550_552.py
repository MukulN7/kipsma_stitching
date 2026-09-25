"""Small diagnostic script to inspect SIFT registration between source frames 550 and 552.

Does NOT modify production code.
Saves frame_550.png and frame_552.png and prints comprehensive SIFT diagnostic metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.registration import RegistrationEngine, RegistrationResult

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260922_18_07_08_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


def main() -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    frame_550 = None
    frame_552 = None

    source_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        source_count += 1
        if source_count == 550:
            frame_550 = cv2.resize(frame, (640, 360))
        elif source_count == 552:
            frame_552 = cv2.resize(frame, (640, 360))

        if source_count >= 552:
            break

    cap.release()

    if frame_550 is None or frame_552 is None:
        print("[ERROR] Could not extract frames 550 and 552.", file=sys.stderr)
        sys.exit(1)

    # Save PNG images
    scratch_dir = project_root / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    f550_path = scratch_dir / "frame_550.png"
    f552_path = scratch_dir / "frame_552.png"

    cv2.imwrite(str(f550_path), frame_550)
    cv2.imwrite(str(f552_path), frame_552)

    engine = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=150.0,
        use_laplacian=True,
        use_sift=True,
    )

    # SIFT feature extraction & matching details
    gray_550 = engine._to_gray_u8(frame_550)
    gray_552 = engine._to_gray_u8(frame_552)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_550, None)
    kp2, des2 = sift.detectAndCompute(gray_552, None)

    num_kp1 = len(kp1) if kp1 is not None else 0
    num_kp2 = len(kp2) if kp2 is not None else 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2) if (des1 is not None and des2 is not None) else []

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    good_count = len(good_matches)

    if good_count >= 6:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    else:
        matrix, inliers = None, None

    if matrix is not None:
        a, b = float(matrix[0, 0]), float(matrix[1, 0])
        scale = float(np.hypot(a, b))
        angle_deg = float(abs(np.degrees(np.arctan2(b, a))))
        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        magnitude = float(np.hypot(dx, dy))
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        inlier_ratio = float(inlier_count / good_count) if good_count > 0 else 0.0
    else:
        scale, angle_deg, dx, dy, magnitude, inlier_count, inlier_ratio = 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

    r_gray = gray_550.astype(np.float32)
    c_gray = gray_552.astype(np.float32)
    spatial_score = float(engine._compute_spatial_overlap_score(r_gray, c_gray, dx, dy))

    # Production RegistrationEngine result
    reg_result: RegistrationResult = engine.register(frame_550, frame_552)

    # Active rejection conditions check
    rejection_conditions = []
    if num_kp1 < 6 or num_kp2 < 6:
        rejection_conditions.append(f"Insufficient keypoints (ref={num_kp1}, curr={num_kp2})")
    if good_count < 6:
        rejection_conditions.append(f"Insufficient good matches ({good_count} < 6)")
    if matrix is None:
        rejection_conditions.append("RANSAC affine matrix estimation failed (matrix is None)")
    else:
        if abs(scale - 1.0) > 0.05:
            rejection_conditions.append(f"Excessive scale change (|{scale:.4f} - 1.0| > 0.05)")
        if angle_deg > 3.0:
            rejection_conditions.append(f"Excessive rotation ({angle_deg:.2f}° > 3.0°)")
        if inlier_count < 10:
            rejection_conditions.append(f"Insufficient inliers ({inlier_count} < 10)")
        if inlier_ratio < 0.45:
            rejection_conditions.append(f"Low inlier ratio ({inlier_ratio:.4f} < 0.45)")
        if engine.max_shift is not None and magnitude > engine.max_shift:
            rejection_conditions.append(f"Max shift exceeded ({magnitude:.2f} px > {engine.max_shift} px)")

    if spatial_score < engine.min_spatial_score:
        rejection_conditions.append(f"Low spatial score ({spatial_score:.4f} < {engine.min_spatial_score})")

    print("=" * 80)
    print("  DIAGNOSTIC ANALYSIS: SOURCE FRAME 550 vs SOURCE FRAME 552")
    print("=" * 80)
    print(f"Saved PNGs               : {f550_path}")
    print(f"                           {f552_path}")
    print("-" * 80)
    print(f"Keypoint Count (Ref 550) : {num_kp1}")
    print(f"Keypoint Count (Curr 552): {num_kp2}")
    print(f"Good Match Count (Lowe)  : {good_count}")
    print(f"RANSAC Inliers Count     : {inlier_count}")
    print(f"Inlier Ratio             : {inlier_ratio:.4f} ({inlier_count}/{good_count})")
    print(f"Displacement Vector (dx) : {dx:.4f} px")
    print(f"Displacement Vector (dy) : {dy:.4f} px")
    print(f"Displacement Magnitude   : {magnitude:.4f} px")
    print(f"Estimated Scale          : {scale:.4f}")
    print(f"Estimated Rotation       : {angle_deg:.4f}°")
    print(f"Spatial Overlap Score    : {spatial_score:.4f}")
    print("-" * 80)
    print(f"Registration Engine Result:")
    print(f"  Valid                  : {reg_result.valid}")
    print(f"  dx                     : {reg_result.dx:.4f} px")
    print(f"  dy                     : {reg_result.dy:.4f} px")
    print(f"  Response (inlier_ratio): {reg_result.response:.4f}")
    print(f"  Spatial Score          : {reg_result.spatial_score:.4f}")
    print("-" * 80)
    print(f"Active Rejection Conditions ({len(rejection_conditions)}):")
    if rejection_conditions:
        for cond in rejection_conditions:
            print(f"  [X] {cond}")
    else:
        print("  [None] Registration is 100% VALID")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
