"""Tests for the horizontal-strip overlap fallback added to RegistrationEngine.

Covers:
  - horizontal translation (rightward)
  - horizontal translation (leftward)
  - vertical translation (full-frame path, not fallback)
  - combined X/Y translation
  - limited-overlap translation detected by the fallback
  - insufficient-overlap rejection (flat texture-free frame)
  - uint8 input
  - uint16 input
  - input immutability

Plus a dataset-conditional diagnostic section for real MIST tile pairs.
"""

import os
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "datasets",
    "Phase_Image_Tiles",
    "Phase_Image_Tiles",
)


def _dataset_available() -> bool:
    return os.path.isdir(DATASET_DIR) and len(
        [f for f in os.listdir(DATASET_DIR) if f.endswith(".tif")]
    ) >= 5


def _make_texture(h, w, dtype=np.uint8):
    """Reproducible random texture with reasonable dynamic range."""
    raw = RNG.integers(30, 220, size=(h, w), dtype=np.uint16)
    if dtype == np.uint16:
        return (raw * 200).astype(np.uint16)
    return raw.astype(np.uint8)


def _crop_shift(frame, dx, dy):
    """Shift frame by (dx, dy) using crop-and-paste into a zero canvas.

    Avoids np.roll circular aliasing that confuses phase-correlation.
    Positive dx means features in frame are shifted RIGHT (+x) in out.
    Positive dy means features in frame are shifted DOWN (+y) in out.
    """
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)

    src_x0 = max(0, -dx);  src_x1 = min(w, w - dx)
    src_y0 = max(0, -dy);  src_y1 = min(h, h - dy)
    dst_x0 = src_x0 + dx;  dst_x1 = src_x1 + dx
    dst_y0 = src_y0 + dy;  dst_y1 = src_y1 + dy

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------


class TestOverlapFallbackSynthetic(unittest.TestCase):

    def setUp(self):
        from microstitch.registration import RegistrationEngine
        self.engine = RegistrationEngine()

    def test_horizontal_right_translation_detected(self):
        """curr_frame recorded to the right of ref_frame -> fallback right hypothesis."""
        frame = _make_texture(200, 300)
        dx_true = 270   # curr tile is 270 px to the right (10 % overlap)
        curr = _crop_shift(frame, -dx_true, 0)
        result = self.engine.register(frame, curr)
        self.assertTrue(result.valid, f"Expected valid=True; got {result}")
        self.assertAlmostEqual(result.dx, dx_true, delta=3.0,
                               msg=f"dx error: {result.dx:.1f} vs {dx_true}")

    def test_horizontal_left_translation_detected(self):
        """curr_frame recorded to the left of ref_frame -> fallback left hypothesis."""
        frame = _make_texture(200, 300)
        dx_true = -270  # curr tile is 270 px to the left
        curr = _crop_shift(frame, -dx_true, 0)
        result = self.engine.register(frame, curr)
        self.assertTrue(result.valid, f"Expected valid=True; got {result}")
        self.assertAlmostEqual(result.dx, dx_true, delta=3.0,
                               msg=f"dx error: {result.dx:.1f} vs {dx_true}")

    def test_vertical_translation_full_frame_path(self):
        """Pure vertical shift should still pass via the full-frame correlation path."""
        frame = _make_texture(200, 300)
        dy_true = 40
        curr = _crop_shift(frame, 0, dy_true)
        result = self.engine.register(frame, curr)
        self.assertTrue(result.valid, f"Expected valid=True; got {result}")
        self.assertAlmostEqual(result.dy, dy_true, delta=3.0,
                               msg=f"dy error: {result.dy:.1f} vs {dy_true}")

    def test_combined_xy_translation(self):
        """Small combined shift recoverable by full-frame phase correlation."""
        frame = _make_texture(200, 300)
        dx_true, dy_true = 20, 15
        curr = _crop_shift(frame, dx_true, dy_true)
        result = self.engine.register(frame, curr)
        self.assertTrue(result.valid, f"Expected valid=True; got {result}")
        self.assertAlmostEqual(result.dx, dx_true, delta=2.0)
        self.assertAlmostEqual(result.dy, dy_true, delta=2.0)

    def test_limited_overlap_10pct_right(self):
        """10 % overlap on right-recorded tile -- canonical fallback scenario."""
        frame = _make_texture(200, 400)
        overlap_px = 40   # 10 % of 400
        dx_true = 400 - overlap_px   # = 360
        curr = _crop_shift(frame, -dx_true, 0)
        result = self.engine.register(frame, curr)
        self.assertTrue(result.valid,
                        f"Expected valid=True for 10 % overlap; got {result}")
        self.assertAlmostEqual(result.dx, dx_true, delta=5.0,
                               msg=f"dx error: {result.dx:.1f} vs {dx_true}")

    def test_flat_frame_returns_invalid(self):
        """Flat (zero-texture) frames should not produce a valid result."""
        flat = np.full((100, 100), 128, dtype=np.uint8)
        result = self.engine.register(flat, flat)
        self.assertFalse(result.valid,
                         "Flat frames should not pass the quality gate")

    def test_uint8_input_accepted(self):
        """uint8 frames must work end-to-end without dtype errors."""
        frame = _make_texture(100, 150, np.uint8)
        curr = _crop_shift(frame, 10, 5)
        result = self.engine.register(frame, curr)
        self.assertTrue(np.isfinite(result.dx))
        self.assertTrue(np.isfinite(result.dy))

    def test_uint16_input_accepted(self):
        """uint16 frames (MIST style) must work end-to-end without dtype errors."""
        frame = _make_texture(100, 150, np.uint16)
        curr = _crop_shift(frame, 10, 5)
        result = self.engine.register(frame, curr)
        self.assertTrue(np.isfinite(result.dx))
        self.assertTrue(np.isfinite(result.dy))

    def test_input_frames_not_modified(self):
        """register() must not modify ref_frame or curr_frame in-place."""
        frame = _make_texture(100, 150)
        curr = _crop_shift(frame, 20, 0)
        ref_copy  = frame.copy()
        curr_copy = curr.copy()
        self.engine.register(frame, curr)
        np.testing.assert_array_equal(frame, ref_copy,
                                      err_msg="ref_frame was modified in-place")
        np.testing.assert_array_equal(curr, curr_copy,
                                      err_msg="curr_frame was modified in-place")


# ---------------------------------------------------------------------------
# MIST dataset diagnostic (skipped when dataset absent)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_dataset_available(), "MIST Phase Image Tiles dataset not found")
class TestMISTHorizontalOverlapDiagnostic(unittest.TestCase):
    """Diagnostic: run the improved engine on first four row-1 adjacent pairs.

    Asserts only that every result is structurally well-formed (finite fields).
    Prints the actual dx/dy/response/spatial/valid values for analysis.
    Does NOT force valid=True -- if the fallback cannot recover the MIST pairs
    reliably, the quality gate must remain intact.
    """

    PAIRS = [
        ("img_Phase_r001_c001.tif", "img_Phase_r001_c002.tif"),
        ("img_Phase_r001_c002.tif", "img_Phase_r001_c003.tif"),
        ("img_Phase_r001_c003.tif", "img_Phase_r001_c004.tif"),
        ("img_Phase_r001_c004.tif", "img_Phase_r001_c005.tif"),
    ]

    def _load(self, filename):
        import tifffile
        return tifffile.imread(os.path.join(DATASET_DIR, filename))

    def setUp(self):
        from microstitch.registration import RegistrationEngine
        self.engine = RegistrationEngine()

    def test_mist_pairs_diagnostic(self):
        from microstitch.registration import RegistrationResult

        # ASCII-only output to avoid cp1252 UnicodeEncodeError on Windows
        print("\n\nMIST row-1 horizontal overlap fallback diagnostic")
        print("=" * 62)
        print(f"{'Pair':<26} {'dx':>8} {'dy':>8} {'resp':>8} {'spatial':>8} {'valid':>6}")
        print("-" * 62)

        for ref_name, curr_name in self.PAIRS:
            ref_frame  = self._load(ref_name)
            curr_frame = self._load(curr_name)
            result = self.engine.register(ref_frame, curr_frame)

            label = f"{ref_name[-11:-4]}->{curr_name[-11:-4]}"
            print(
                f"{label:<26} {result.dx:>8.2f} {result.dy:>8.2f} "
                f"{result.response:>8.4f} {result.spatial_score:>8.4f} "
                f"{str(result.valid):>6}"
            )

            self.assertIsInstance(result, RegistrationResult)
            self.assertTrue(np.isfinite(result.dx),            f"dx not finite: {ref_name}")
            self.assertTrue(np.isfinite(result.dy),            f"dy not finite: {ref_name}")
            self.assertTrue(np.isfinite(result.response),      f"response not finite: {ref_name}")
            self.assertTrue(np.isfinite(result.spatial_score), f"spatial_score not finite: {ref_name}")

        print("=" * 62)


if __name__ == "__main__":
    unittest.main(verbosity=2)
