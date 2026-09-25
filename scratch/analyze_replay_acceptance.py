"""Offline video replay registration acceptance analysis script.

Analyzes why frame registrations are accepted or rejected during video replay.
Does NOT modify production code, thresholds, tests, or replay_video.py.

Output:
    outputs/replay_acceptance_analysis.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from microstitch.movement_detector import MovementDetector
from microstitch.registration import RegistrationEngine, RegistrationResult
from microstitch.mosaic import MosaicManager

DEFAULT_VIDEO_PATH = r"C:\Users\mukul\OneDrive\Pictures\Camera Roll 1\WIN_20260922_18_07_08_Pro.mp4"
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH


def analyze_sift(engine: RegistrationEngine, ref_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, Any]:
    """Extracts detailed SIFT feature matching metrics for diagnostic analysis."""
    gray_ref = engine._to_gray_u8(ref_frame)
    gray_curr = engine._to_gray_u8(curr_frame)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

    num_kp_ref = len(kp_ref) if kp_ref is not None else 0
    num_kp_curr = len(kp_curr) if kp_curr is not None else 0

    if des_ref is None or des_curr is None or num_kp_ref < 6 or num_kp_curr < 6:
        return {
            "sift_valid": False,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "scale": 1.0,
            "angle_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "spatial_score": 0.0,
            "reasons": ["insufficient matches"],
            "matrix_found": False,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_curr, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    good_match_count = len(good_matches)
    if good_match_count < 6:
        return {
            "sift_valid": False,
            "good_matches": good_match_count,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "scale": 1.0,
            "angle_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "spatial_score": 0.0,
            "reasons": ["insufficient matches"],
            "matrix_found": False,
        }

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )

    if matrix is None:
        return {
            "sift_valid": False,
            "good_matches": good_match_count,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "scale": 1.0,
            "angle_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "magnitude": 0.0,
            "spatial_score": 0.0,
            "reasons": ["insufficient inliers"],
            "matrix_found": False,
        }

    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = float(np.hypot(a, b))
    angle_deg = float(abs(np.degrees(np.arctan2(b, a))))
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    magnitude = float(np.hypot(dx, dy))

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    inlier_ratio = float(inlier_count / good_match_count) if good_match_count > 0 else 0.0

    r_gray = gray_ref.astype(np.float32)
    c_gray = gray_curr.astype(np.float32)
    spatial_score = float(engine._compute_spatial_overlap_score(r_gray, c_gray, dx, dy))

    reasons = []
    if good_match_count < 6:
        reasons.append("insufficient matches")
    if inlier_count < 10:
        reasons.append("insufficient inliers")
    if inlier_ratio < 0.45:
        reasons.append("low inlier ratio")
    if angle_deg > 3.0:
        reasons.append("excessive rotation")
    if abs(scale - 1.0) > 0.05:
        reasons.append("excessive scale change")
    if engine.max_shift is not None and magnitude > engine.max_shift:
        reasons.append("max_shift")
    if spatial_score < engine.min_spatial_score:
        reasons.append("low spatial score")

    sift_valid = bool(
        abs(scale - 1.0) <= 0.05 and
        angle_deg <= 3.0 and
        inlier_count >= 10 and
        inlier_ratio >= 0.45 and
        (engine.max_shift is None or magnitude <= engine.max_shift) and
        np.isfinite(dx) and np.isfinite(dy)
    )

    return {
        "sift_valid": sift_valid,
        "good_matches": good_match_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "scale": scale,
        "angle_deg": angle_deg,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "spatial_score": spatial_score,
        "reasons": reasons,
        "matrix_found": True,
    }


def determine_rejection_reason(reg: RegistrationResult, sift_diag: Dict[str, Any], engine: RegistrationEngine) -> str:
    """Determines a concise, human-readable primary rejection reason for a failed registration."""
    if reg.valid:
        return "ACCEPTED"

    sift_reasons = sift_diag.get("reasons", [])

    if "insufficient matches" in sift_reasons:
        return "SIFT insufficient matches (< 6)"
    if "insufficient inliers" in sift_reasons:
        return f"SIFT insufficient inliers ({sift_diag['inliers']} < 10)"
    if "low inlier ratio" in sift_reasons:
        return f"SIFT low inlier ratio ({sift_diag['inlier_ratio']:.2f} < 0.45)"
    if "excessive rotation" in sift_reasons:
        return f"SIFT excessive rotation ({sift_diag['angle_deg']:.1f}° > 3.0°)"
    if "excessive scale change" in sift_reasons:
        return f"SIFT excessive scale change (|{sift_diag['scale']:.2f}-1| > 0.05)"
    if "max_shift" in sift_reasons:
        return f"SIFT max shift exceeded ({sift_diag['magnitude']:.1f}px > 150px)"
    if "low spatial score" in sift_reasons:
        return f"SIFT low spatial score ({sift_diag['spatial_score']:.2f} < 0.35)"

    phase_reasons = []
    if reg.response < engine.min_response:
        phase_reasons.append(f"Phase low response ({reg.response:.4f} < {engine.min_response})")
    if reg.spatial_score < engine.min_spatial_score:
        phase_reasons.append(f"Phase low spatial score ({reg.spatial_score:.4f} < {engine.min_spatial_score})")
    mag = np.hypot(reg.dx, reg.dy)
    if engine.max_shift is not None and mag > engine.max_shift:
        phase_reasons.append(f"Phase max shift exceeded ({mag:.1f}px > 150px)")

    if phase_reasons:
        return "; ".join(phase_reasons)

    return "Registration failed quality gate"


def print_distribution(name: str, records: List[Dict[str, Any]]) -> None:
    """Prints count, medians, and percentiles for a subset of records."""
    count = len(records)
    print(f"\n--- {name} (Count: {count}) ---")
    if count == 0:
        print("  No records.")
        return

    mags = [r["magnitude"] for r in records]
    resps = [r["response"] for r in records]
    spatials = [r["spatial_score"] for r in records]
    inlier_ratios = [r["sift_inlier_ratio"] for r in records]

    med_mag = float(np.median(mags))
    p90_mag = float(np.percentile(mags, 90))
    med_resp = float(np.median(resps))
    med_spatial = float(np.median(spatials))
    med_inlier_ratio = float(np.median(inlier_ratios))

    print(f"  Count                : {count}")
    print(f"  Median Magnitude     : {med_mag:.2f} px")
    print(f"  P90 Magnitude        : {p90_mag:.2f} px")
    print(f"  Median Response      : {med_resp:.4f}")
    print(f"  Median Spatial Score : {med_spatial:.4f}")
    print(f"  Median Inlier Ratio  : {med_inlier_ratio:.4f}")


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
    mosaic = MosaicManager()

    source_frame_count = 0
    processed_count = 0
    prev_accepted: np.ndarray | None = None

    records: List[Dict[str, Any]] = []
    rejected_sift_failures: List[Dict[str, Any]] = []

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

        moved, mad = movement.process_frame(frame)
        if not moved:
            continue

        if prev_accepted is None:
            mosaic.add_first_frame(frame)
            prev_accepted = frame.copy()
            continue

        sift_diag = analyze_sift(engine, prev_accepted, frame)
        reg: RegistrationResult = engine.register(prev_accepted, frame)

        is_valid = reg.valid
        magnitude = float(np.hypot(reg.dx, reg.dy))
        rej_reason = determine_rejection_reason(reg, sift_diag, engine) if not is_valid else ""

        if is_valid:
            tile = mosaic.add_frame(frame, reg)
            if tile is not None:
                prev_accepted = frame.copy()
            else:
                is_valid = False
                rej_reason = "Rejected by MosaicManager"

        record = {
            "source_frame": source_frame_count,
            "mad": mad,
            "dx": reg.dx,
            "dy": reg.dy,
            "magnitude": magnitude,
            "response": reg.response,
            "spatial_score": reg.spatial_score,
            "valid": is_valid,
            "rejection_reason": rej_reason,
            "sift_good_matches": sift_diag["good_matches"],
            "sift_inliers": sift_diag["inliers"],
            "sift_inlier_ratio": sift_diag["inlier_ratio"],
            "sift_diag": sift_diag,
            "reg": reg,
        }
        records.append(record)

        if not sift_diag["sift_valid"]:
            sift_failure_info = {
                "source_frame": source_frame_count,
                "sift_diag": sift_diag,
                "reg_valid": is_valid,
                "reg": reg,
            }
            rejected_sift_failures.append(sift_failure_info)

    cap.release()
    elapsed = time.time() - start_time

    # Save CSV output
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "replay_acceptance_analysis.csv"

    fieldnames = [
        "source_frame", "mad", "dx", "dy", "magnitude", "response",
        "spatial_score", "valid", "rejection_reason",
        "sift_good_matches", "sift_inliers", "sift_inlier_ratio"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "source_frame": r["source_frame"],
                "mad": round(r["mad"], 4),
                "dx": round(r["dx"], 4),
                "dy": round(r["dy"], 4),
                "magnitude": round(r["magnitude"], 4),
                "response": round(r["response"], 4),
                "spatial_score": round(r["spatial_score"], 4),
                "valid": r["valid"],
                "rejection_reason": r["rejection_reason"],
                "sift_good_matches": r["sift_good_matches"],
                "sift_inliers": r["sift_inliers"],
                "sift_inlier_ratio": round(r["sift_inlier_ratio"], 4),
            }
            writer.writerow(row)
        f.flush()

    print("=" * 80)
    print("  MICROSCOPE REPLAY REGISTRATION ACCEPTANCE ANALYSIS")
    print("=" * 80)
    print(f"Total source frames evaluated     : {source_frame_count}")
    print(f"Processed frames (every 5th)      : {processed_count}")
    print(f"Frames reaching RegistrationEngine: {len(records)}")
    print(f"Accepted registrations            : {sum(1 for r in records if r['valid'])}")
    print(f"Rejected registrations            : {sum(1 for r in records if not r['valid'])}")
    print(f"Analysis saved to                 : {csv_path.resolve()}")
    print("=" * 80)

    # 1. Summary grouped by rejection reason
    rejected_records = [r for r in records if not r["valid"]]
    reason_counts: Dict[str, int] = {}
    for r in rejected_records:
        reason = r["rejection_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("\n==========================================================================")
    print(" 1. REJECTION REASON BREAKDOWN")
    print("==========================================================================")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / len(rejected_records) if rejected_records else 0.0
        print(f"  {reason:<55}: {count:>4} ({pct:>5.1f}%)")

    # 2. Distributions
    print("\n==========================================================================")
    print(" 2. PARAMETER DISTRIBUTIONS")
    print("==========================================================================")
    accepted_records = [r for r in records if r["valid"]]
    print_distribution("ACCEPTED REGISTRATIONS", accepted_records)
    print_distribution("REJECTED REGISTRATIONS", rejected_records)

    # 3. SIFT condition failure counts for rejected SIFT results
    print("\n==========================================================================")
    print(" 3. REJECTED SIFT CONDITION FAILURES")
    print("==========================================================================")
    print(f"Total rejected SIFT results: {len(rejected_sift_failures)}")

    cond_counts = {
        "insufficient matches": 0,
        "insufficient inliers": 0,
        "low inlier ratio": 0,
        "excessive rotation": 0,
        "excessive scale change": 0,
        "max_shift": 0,
        "low spatial score": 0,
        "phase fallback failure": 0,
    }

    for sf in rejected_sift_failures:
        diag = sf["sift_diag"]
        reasons = diag.get("reasons", [])

        if "insufficient matches" in reasons:
            cond_counts["insufficient matches"] += 1
        if "insufficient inliers" in reasons:
            cond_counts["insufficient inliers"] += 1
        if "low inlier ratio" in reasons:
            cond_counts["low inlier ratio"] += 1
        if "excessive rotation" in reasons:
            cond_counts["excessive rotation"] += 1
        if "excessive scale change" in reasons:
            cond_counts["excessive scale change"] += 1
        if "max_shift" in reasons:
            cond_counts["max_shift"] += 1
        if "low spatial score" in reasons:
            cond_counts["low spatial score"] += 1
        if not sf["reg_valid"]:
            cond_counts["phase fallback failure"] += 1

    for cond, count in cond_counts.items():
        pct = 100.0 * count / len(rejected_sift_failures) if rejected_sift_failures else 0.0
        print(f"  {cond:<25}: {count:>4} ({pct:>5.1f}%)")

    # 4. Specific subsets of rejected registrations
    print("\n==========================================================================")
    print(" 4. SPECIFIC REJECTED REGISTRATION SUBSETS")
    print("==========================================================================")
    print(f"Total rejected registrations: {len(rejected_records)}")

    inlier_ge_50 = sum(1 for r in rejected_records if r["sift_inlier_ratio"] >= 0.50)
    inlier_ge_70 = sum(1 for r in rejected_records if r["sift_inlier_ratio"] >= 0.70)
    inlier_ge_80 = sum(1 for r in rejected_records if r["sift_inlier_ratio"] >= 0.80)
    mag_le_150 = sum(1 for r in rejected_records if r["magnitude"] <= 150.0)
    spatial_lt_35 = sum(1 for r in rejected_records if r["spatial_score"] < 0.35)

    print(f"  Inlier ratio >= 0.50 : {inlier_ge_50:>4} ({100.0*inlier_ge_50/max(len(rejected_records),1):.1f}%)")
    print(f"  Inlier ratio >= 0.70 : {inlier_ge_70:>4} ({100.0*inlier_ge_70/max(len(rejected_records),1):.1f}%)")
    print(f"  Inlier ratio >= 0.80 : {inlier_ge_80:>4} ({100.0*inlier_ge_80/max(len(rejected_records),1):.1f}%)")
    print(f"  Magnitude <= 150 px  : {mag_le_150:>4} ({100.0*mag_le_150/max(len(rejected_records),1):.1f}%)")
    print(f"  Spatial score < 0.35 : {spatial_lt_35:>4} ({100.0*spatial_lt_35/max(len(rejected_records),1):.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
