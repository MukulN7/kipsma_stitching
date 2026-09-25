"""Lightweight diagnostic script for SIFT row-transition failure analysis.

Stops at the FIRST SIFT rejection after frame 550, prints full diagnostic metrics
for the rejected frame and the next 5 frames, then exits cleanly.

Does NOT modify production code.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.movement_detector import MovementDetector
from microstitch.registration import RegistrationEngine, RegistrationResult

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260922_18_07_08_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


def extract_sift_metrics(engine: RegistrationEngine, ref_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, Any]:
    """Extracts detailed SIFT feature matching and validation metrics for a frame pair."""
    gray_ref = engine._to_gray_u8(ref_frame)
    gray_curr = engine._to_gray_u8(curr_frame)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

    num_kp_ref = len(kp_ref) if kp_ref is not None else 0
    num_kp_curr = len(kp_curr) if kp_curr is not None else 0

    if des_ref is None or des_curr is None or num_kp_ref < 6 or num_kp_curr < 6:
        return {
            "kp_ref": num_kp_ref, "kp_curr": num_kp_curr,
            "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "dx": 0.0, "dy": 0.0, "magnitude": 0.0,
            "scale": 1.0, "rotation_deg": 0.0, "spatial_score": 0.0,
            "reasons": [f"Insufficient keypoints (ref={num_kp_ref}, curr={num_kp_curr})"]
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_curr, k=2)

    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    good_count = len(good_matches)

    if good_count < 6:
        return {
            "kp_ref": num_kp_ref, "kp_curr": num_kp_curr,
            "good_matches": good_count, "inliers": 0, "inlier_ratio": 0.0,
            "dx": 0.0, "dy": 0.0, "magnitude": 0.0,
            "scale": 1.0, "rotation_deg": 0.0, "spatial_score": 0.0,
            "reasons": [f"Insufficient good matches ({good_count} < 6)"]
        }

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if matrix is None:
        return {
            "kp_ref": num_kp_ref, "kp_curr": num_kp_curr,
            "good_matches": good_count, "inliers": 0, "inlier_ratio": 0.0,
            "dx": 0.0, "dy": 0.0, "magnitude": 0.0,
            "scale": 1.0, "rotation_deg": 0.0, "spatial_score": 0.0,
            "reasons": ["RANSAC affine estimation failed"]
        }

    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = float(np.hypot(a, b))
    rotation_deg = float(abs(np.degrees(np.arctan2(b, a))))
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    magnitude = float(np.hypot(dx, dy))

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    inlier_ratio = float(inlier_count / good_count) if good_count > 0 else 0.0

    r_gray = gray_ref.astype(np.float32)
    c_gray = gray_curr.astype(np.float32)
    spatial_score = float(engine._compute_spatial_overlap_score(r_gray, c_gray, dx, dy))

    reasons = []
    if abs(scale - 1.0) > 0.05:
        reasons.append(f"Excessive scale change (|{scale:.4f} - 1.0| > 0.05)")
    if rotation_deg > 3.0:
        reasons.append(f"Excessive rotation ({rotation_deg:.2f}° > 3.0°)")
    if inlier_count < 5:
        reasons.append(f"Insufficient inliers ({inlier_count} < 5)")
    if inlier_ratio < 0.45:
        reasons.append(f"Low inlier ratio ({inlier_ratio:.4f} < 0.45)")
    if inlier_count < 10 and inlier_ratio < 0.50:
        reasons.append(f"Low inlier count & ratio ({inlier_count} < 10 and ratio {inlier_ratio:.4f} < 0.50)")
    if engine.max_shift is not None and magnitude > engine.max_shift:
        reasons.append(f"Max shift exceeded ({magnitude:.2f}px > {engine.max_shift}px)")
    if spatial_score < engine.min_spatial_score:
        reasons.append(f"Low spatial score ({spatial_score:.4f} < {engine.min_spatial_score})")

    return {
        "kp_ref": num_kp_ref,
        "kp_curr": num_kp_curr,
        "good_matches": good_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "spatial_score": spatial_score,
        "reasons": reasons,
    }


def print_diagnostic_block(tag: str, source_frame: int, prev_accepted: int, metrics: Dict[str, Any], reg: RegistrationResult) -> None:
    gap = source_frame - prev_accepted
    reasons = metrics["reasons"] if metrics["reasons"] else ["None (VALID)"]
    reason_str = " | ".join(reasons)

    print(f"\n[{tag}] Candidate Frame: {source_frame} | Ref Frame: {prev_accepted} | Gap: {gap} frames")
    print(f"  • Keypoints           : Ref={metrics['kp_ref']} | Curr={metrics['kp_curr']}")
    print(f"  • Good Matches        : {metrics['good_matches']}")
    print(f"  • RANSAC Inliers      : {metrics['inliers']} (Ratio: {metrics['inlier_ratio']:.4f})")
    print(f"  • Displacement Vector : dx={metrics['dx']:.4f} px | dy={metrics['dy']:.4f} px")
    print(f"  • Magnitude           : {metrics['magnitude']:.4f} px")
    print(f"  • Scale & Rotation    : Scale={metrics['scale']:.4f} | Rotation={metrics['rotation_deg']:.4f}°")
    print(f"  • Spatial Overlap     : {metrics['spatial_score']:.4f}")
    print(f"  • Engine Valid Status : {reg.valid}")
    print(f"  • Rejection Reason    : {reason_str}")


def main() -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    movement = MovementDetector(threshold=3.0, auto_update_reference=True)
    engine = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=150.0,
        use_laplacian=True,
        use_sift=True,
    )

    source_frame_count = 0
    candidate_count = 0

    prev_accepted_idx: int | None = None
    prev_accepted_img: np.ndarray | None = None

    first_rejected_source_idx: int | None = None
    post_rejection_eval_count = 0

    print("=" * 95)
    print("  SIFT ROW-TRANSITION DIAGNOSTIC (Stopping after first rejection post-550 + next 5 frames)")
    print("=" * 95)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        source_frame_count += 1
        frame = cv2.resize(frame, (640, 360))

        moved, mad = movement.process_frame(frame)
        if not moved:
            continue

        candidate_count += 1

        if prev_accepted_img is None:
            prev_accepted_idx = source_frame_count
            prev_accepted_img = frame.copy()
            continue

        reg = engine.register(prev_accepted_img, frame)

        # Before frame 550, update reference if valid
        if first_rejected_source_idx is None:
            if reg.valid:
                prev_accepted_idx = source_frame_count
                prev_accepted_img = frame.copy()
            else:
                if source_frame_count > 550:
                    first_rejected_source_idx = source_frame_count
                    metrics = extract_sift_metrics(engine, prev_accepted_img, frame)
                    print_diagnostic_block("FIRST REJECTED FRAME AFTER 550", source_frame_count, prev_accepted_idx, metrics, reg)
                    post_rejection_eval_count = 0
        else:
            # Evaluate the next 5 frames after first rejection
            post_rejection_eval_count += 1
            metrics = extract_sift_metrics(engine, prev_accepted_img, frame)
            print_diagnostic_block(f"NEXT FRAME +{post_rejection_eval_count} AFTER REJECTION", source_frame_count, prev_accepted_idx, metrics, reg)

            if post_rejection_eval_count >= 5:
                break

    cap.release()
    print("\n" + "=" * 95)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    main()
