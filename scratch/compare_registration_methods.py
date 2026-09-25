"""Diagnostic comparison script between Phase Correlation and SIFT + RANSAC.

Evaluates 100 representative adjacent frame pairs from a recorded microscope video
to compare existing RegistrationEngine (phase correlation) against SIFT feature matching.

Usage:
    python scratch/compare_registration_methods.py [video_path]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.registration import RegistrationEngine, RegistrationResult

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260922_18_07_08_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


def run_sift_registration(frame_a: np.ndarray, frame_b: np.ndarray) -> dict:
    """Computes 2D translation via SIFT feature detection and RANSAC affine estimation."""
    if frame_a.ndim == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    else:
        gray_a = frame_a
    if frame_b.ndim == 3:
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_b = frame_b

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 6 or len(kp_b) < 6:
        return {
            "sift_dx": 0.0,
            "sift_dy": 0.0,
            "sift_magnitude": 0.0,
            "sift_good_matches": 0,
            "sift_inliers": 0,
            "sift_inlier_ratio": 0.0,
            "sift_valid": False,
            "sift_le_150": False,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_a, des_b, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 6:
        return {
            "sift_dx": 0.0,
            "sift_dy": 0.0,
            "sift_magnitude": 0.0,
            "sift_good_matches": len(good_matches),
            "sift_inliers": 0,
            "sift_inlier_ratio": 0.0,
            "sift_valid": False,
            "sift_le_150": False,
        }

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )

    if matrix is None:
        return {
            "sift_dx": 0.0,
            "sift_dy": 0.0,
            "sift_magnitude": 0.0,
            "sift_good_matches": len(good_matches),
            "sift_inliers": 0,
            "sift_inlier_ratio": 0.0,
            "sift_valid": False,
            "sift_le_150": False,
        }

    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = float(np.hypot(a, b))
    angle_deg = float(abs(np.degrees(np.arctan2(b, a))))
    tx = float(matrix[0, 2])
    ty = float(matrix[1, 2])
    mag = float(np.hypot(tx, ty))

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    inlier_ratio = float(inlier_count / len(good_matches))

    # Reject if matrix contains meaningful scale change or rotation
    valid = bool(
        (abs(scale - 1.0) <= 0.05) and
        (angle_deg <= 3.0) and
        (inlier_count >= 6) and
        (inlier_ratio >= 0.35)
    )

    return {
        "sift_dx": tx,
        "sift_dy": ty,
        "sift_magnitude": mag,
        "sift_good_matches": len(good_matches),
        "sift_inliers": inlier_count,
        "sift_inlier_ratio": inlier_ratio,
        "sift_valid": valid,
        "sift_le_150": (mag <= 150.0) if valid else False,
    }


def main() -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading video: {video_path}")
    t0 = time.time()
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_resized = cv2.resize(frame, (640, 360))
        frames.append(frame_resized)
    cap.release()

    total_frames = len(frames)
    print(f"Loaded {total_frames} frames resized to 640x360 in {time.time() - t0:.2f}s\n")

    engine = RegistrationEngine(
        min_response=0.05,
        min_spatial_score=0.35,
        max_shift=None,
        use_laplacian=True,
    )

    # 1. Scan adjacent pairs to find suspicious pairs (>200 px magnitude)
    print("Scanning adjacent frame pairs for suspicious phase-correlation shifts (>200 px)...")
    all_pairs: list[tuple[int, int, RegistrationResult]] = []
    suspicious_pairs: list[tuple[int, int, RegistrationResult]] = []
    normal_pairs: list[tuple[int, int, RegistrationResult]] = []

    for i in range(total_frames - 1):
        j = i + 1
        reg_res = engine.register(frames[i], frames[j])
        mag = float(np.hypot(reg_res.dx, reg_res.dy))
        item = (i, j, reg_res)
        all_pairs.append(item)
        if mag > 200.0:
            suspicious_pairs.append(item)
        else:
            normal_pairs.append(item)

    print(f"Found {len(suspicious_pairs)} suspicious pairs (>200 px) out of {len(all_pairs)} total pairs.")

    # Select 100 representative pairs, prioritizing suspicious pairs
    selected_pairs: list[tuple[int, int, RegistrationResult]] = []
    if len(suspicious_pairs) >= 100:
        step = len(suspicious_pairs) / 100.0
        selected_pairs = [suspicious_pairs[int(i * step)] for i in range(100)]
    else:
        selected_pairs.extend(suspicious_pairs)
        remaining_needed = 100 - len(selected_pairs)
        if normal_pairs:
            step = len(normal_pairs) / float(remaining_needed)
            selected_pairs.extend([normal_pairs[int(i * step)] for i in range(remaining_needed)])

    selected_pairs = selected_pairs[:100]
    print(f"Selected {len(selected_pairs)} representative pairs for detailed comparison.\n")

    # 2. Run detailed comparison on the selected pairs
    results: list[dict] = []
    for i, j, phase_res in selected_pairs:
        frame_a = frames[i]
        frame_b = frames[j]

        phase_dx = float(phase_res.dx)
        phase_dy = float(phase_res.dy)
        phase_mag = float(np.hypot(phase_dx, phase_dy))

        sift_res = run_sift_registration(frame_a, frame_b)

        rec = {
            "frame_a": i,
            "frame_b": j,
            "phase_dx": phase_dx,
            "phase_dy": phase_dy,
            "phase_magnitude": phase_mag,
            "phase_response": float(phase_res.response),
            "phase_spatial": float(phase_res.spatial_score),
            "phase_valid": bool(phase_res.valid),
            "sift_dx": sift_res["sift_dx"],
            "sift_dy": sift_res["sift_dy"],
            "sift_magnitude": sift_res["sift_magnitude"],
            "sift_good_matches": sift_res["sift_good_matches"],
            "sift_inliers": sift_res["sift_inliers"],
            "sift_inlier_ratio": sift_res["sift_inlier_ratio"],
            "sift_valid": sift_res["sift_valid"],
            "sift_le_150": sift_res["sift_le_150"],
        }
        results.append(rec)

    # Save CSV
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "registration_method_comparison.csv"

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved CSV comparison to: {csv_path}")

    # Summary metrics
    n_tested = len(results)
    phase_gt_200 = sum(1 for r in results if r["phase_magnitude"] > 200.0)
    sift_valid_count = sum(1 for r in results if r["sift_valid"])
    sift_valid_pct = (100.0 * sift_valid_count / n_tested) if n_tested > 0 else 0.0

    valid_sift_mags = [r["sift_magnitude"] for r in results if r["sift_valid"]]
    all_sift_mags = [r["sift_magnitude"] for r in results]
    sift_mags_to_use = valid_sift_mags if valid_sift_mags else all_sift_mags

    med_sift_mag = float(np.median(sift_mags_to_use)) if sift_mags_to_use else 0.0
    p90_sift_mag = float(np.percentile(sift_mags_to_use, 90)) if sift_mags_to_use else 0.0

    good_matches_list = [r["sift_good_matches"] for r in results]
    inliers_list = [r["sift_inliers"] for r in results]
    inlier_ratios_list = [r["sift_inlier_ratio"] for r in results]

    med_good_matches = float(np.median(good_matches_list)) if good_matches_list else 0.0
    med_inliers = float(np.median(inliers_list)) if inliers_list else 0.0
    med_inlier_ratio = float(np.median(inlier_ratios_list)) if inlier_ratios_list else 0.0

    sift_le_150_count = sum(1 for r in results if r["sift_valid"] and r["sift_magnitude"] <= 150.0)
    phase_gt_200_and_sift_le_150 = sum(
        1 for r in results if r["phase_magnitude"] > 200.0 and r["sift_valid"] and r["sift_magnitude"] <= 150.0
    )

    print("\n" + "=" * 80)
    print("  REGISTRATION METHOD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"  Number of pairs tested                          : {n_tested}")
    print(f"  Phase results with magnitude > 200 px           : {phase_gt_200}")
    print(f"  SIFT valid percentage                           : {sift_valid_pct:.1f}% ({sift_valid_count}/{n_tested})")
    print(f"  Median SIFT magnitude                           : {med_sift_mag:.2f} px")
    print(f"  90th percentile SIFT magnitude                  : {p90_sift_mag:.2f} px")
    print(f"  Median SIFT good matches                        : {med_good_matches:.1f}")
    print(f"  Median SIFT inlier count                        : {med_inliers:.1f}")
    print(f"  Median SIFT inlier ratio                        : {med_inlier_ratio:.4f}")
    print(f"  Number of SIFT results <= 150 px                : {sift_le_150_count}")
    print(f"  Pairs with Phase > 200 px BUT SIFT <= 150 px     : {phase_gt_200_and_sift_le_150}")
    print("=" * 80)

    # Print 20 representative rows
    step_display = max(1, len(results) // 20)
    sample_rows = results[::step_display][:20]

    print("\n" + "=" * 125)
    print(f"  SAMPLE 20 REPRESENTATIVE PAIR COMPARISONS")
    print("=" * 125)
    print(f"{'Pair':>10} | {'Phase dx/dy':>18} | {'Phase Mag':>10} | {'Resp/Spat':>16} | "
          f"{'SIFT dx/dy':>18} | {'SIFT Mag':>10} | {'Matches':>8} | {'Inliers':>8} | {'SIFT Valid':>10}")
    print("-" * 125)

    for r in sample_rows:
        pair_str = f"{r['frame_a']:>4}->{r['frame_b']:<5}"
        p_xy_str = f"({r['phase_dx']:>7.1f},{r['phase_dy']:>7.1f})"
        p_score_str = f"{r['phase_response']:>.3f}/{r['phase_spatial']:>.3f}"
        s_xy_str = f"({r['sift_dx']:>7.1f},{r['sift_dy']:>7.1f})" if r["sift_valid"] else "(       N/A       )"
        s_mag_str = f"{r['sift_magnitude']:>10.2f}" if r["sift_valid"] else "       N/A"
        s_v_str = "VALID" if r["sift_valid"] else "INVALID"

        print(f"{pair_str} | {p_xy_str:>18} | {r['phase_magnitude']:>10.2f} | {p_score_str:>16} | "
              f"{s_xy_str:>18} | {s_mag_str:>10} | {r['sift_good_matches']:>8} | {r['sift_inliers']:>8} | {s_v_str:>10}")
    print("=" * 125)


if __name__ == "__main__":
    main()
