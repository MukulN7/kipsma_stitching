"""Temporal gap registration diagnostic for recorded microscope video.

Evaluates RegistrationEngine performance across different temporal frame gaps
(1, 2, 3, 5, 8 frames) to determine the optimal sampling interval for video stitching.

Usage:
    python scratch/analyze_video_registration.py [video_path]
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

GAPS = [1, 2, 3, 5, 8]
MAX_PAIRS_PER_GAP = 500


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

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "video_registration_analysis.csv"

    all_records: list[dict] = []
    gap_summaries: list[dict] = []
    gap_first10: dict[int, list[dict]] = {}

    for gap in GAPS:
        print(f"Analyzing temporal gap = {gap} frames...")
        max_pairs = min(MAX_PAIRS_PER_GAP, total_frames - gap)
        if max_pairs <= 0:
            print(f"  [WARN] Not enough frames for gap {gap}")
            continue

        gap_records: list[dict] = []

        for i in range(max_pairs):
            j = i + gap
            ref_f = frames[i]
            curr_f = frames[j]

            f_res: RegistrationResult = engine.register(ref_f, curr_f)
            r_res: RegistrationResult = engine.register(curr_f, ref_f)

            f_dx, f_dy = f_res.dx, f_res.dy
            r_dx, r_dy = r_res.dx, r_res.dy
            mag = float(np.hypot(f_dx, f_dy))
            canc_err = float(np.hypot(f_dx + r_dx, f_dy + r_dy))

            rec = {
                "gap": gap,
                "ref_idx": i,
                "curr_idx": j,
                "forward_dx": f_dx,
                "forward_dy": f_dy,
                "magnitude": mag,
                "forward_response": f_res.response,
                "forward_spatial": f_res.spatial_score,
                "forward_valid": f_res.valid,
                "reverse_dx": r_dx,
                "reverse_dy": r_dy,
                "reverse_response": r_res.response,
                "reverse_spatial": r_res.spatial_score,
                "reverse_valid": r_res.valid,
                "cancellation_error": canc_err,
            }
            gap_records.append(rec)
            all_records.append(rec)

        gap_first10[gap] = gap_records[:10]

        # Calculate metrics for summary table
        n_pairs = len(gap_records)
        n_valid = sum(1 for r in gap_records if r["forward_valid"])
        valid_pct = (100.0 * n_valid / n_pairs) if n_pairs > 0 else 0.0

        mags = [r["magnitude"] for r in gap_records]
        resps = [r["forward_response"] for r in gap_records]
        spats = [r["forward_spatial"] for r in gap_records]
        cancs = [r["cancellation_error"] for r in gap_records]

        med_mag = float(np.median(mags)) if mags else 0.0
        p90_mag = float(np.percentile(mags, 90)) if mags else 0.0
        med_resp = float(np.median(resps)) if resps else 0.0
        med_spat = float(np.median(spats)) if spats else 0.0
        med_canc = float(np.median(cancs)) if cancs else 0.0

        gt_200 = sum(1 for m in mags if m > 200.0)
        gt_300 = sum(1 for m in mags if m > 300.0)
        gt_500 = sum(1 for m in mags if m > 500.0)

        gap_summaries.append({
            "gap": gap,
            "pair_count": n_pairs,
            "valid_pct": valid_pct,
            "med_magnitude": med_mag,
            "p90_magnitude": p90_mag,
            "med_response": med_resp,
            "med_spatial": med_spat,
            "med_cancellation": med_canc,
            "gt_200": gt_200,
            "gt_300": gt_300,
            "gt_500": gt_500,
        })

    # Save CSV
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nSaved CSV analysis to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 115)
    print("  TEMPORAL GAP REGISTRATION ANALYSIS SUMMARY")
    print("=" * 115)
    print(f"{'Gap':>4} | {'Pairs':>6} | {'Valid %':>8} | {'Med Mag':>9} | {'P90 Mag':>9} | "
          f"{'Med Resp':>9} | {'Med Spat':>9} | {'Med Canc':>9} | {'>200px':>6} | {'>300px':>6} | {'>500px':>6}")
    print("-" * 115)

    for s in gap_summaries:
        print(f"{s['gap']:>4} | {s['pair_count']:>6} | {s['valid_pct']:>7.1f}% | {s['med_magnitude']:>9.2f} | "
              f"{s['p90_magnitude']:>9.2f} | {s['med_response']:>9.4f} | {s['med_spatial']:>9.4f} | "
              f"{s['med_cancellation']:>9.2f} | {s['gt_200']:>6} | {s['gt_300']:>6} | {s['gt_500']:>6}")
    print("=" * 115)

    # Print first 10 measurements per gap
    for gap in GAPS:
        first10 = gap_first10.get(gap, [])
        print("\n" + "=" * 115)
        print(f"  FIRST {len(first10)} MEASUREMENTS FOR GAP = {gap}")
        print("=" * 115)
        print(f"{'Ref->Curr':>10} | {'dx':>8} | {'dy':>8} | {'Mag':>8} | {'Resp':>7} | {'Spat':>7} | {'Valid':>6} | {'CancErr':>8}")
        print("-" * 115)
        for r in first10:
            v_str = "OK" if r["forward_valid"] else "REJECT"
            print(f"{r['ref_idx']:>4}->{r['curr_idx']:<5} | {r['forward_dx']:>8.2f} | {r['forward_dy']:>8.2f} | "
                  f"{r['magnitude']:>8.2f} | {r['forward_response']:>7.4f} | {r['forward_spatial']:>7.4f} | "
                  f"{v_str:>6} | {r['cancellation_error']:>8.2f}")
        print("=" * 115)


if __name__ == "__main__":
    main()
