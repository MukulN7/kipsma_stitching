import unittest
import cv2
import numpy as np
from microstitch.registration import RegistrationEngine, RegistrationResult


class TestRegistrationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RegistrationEngine(min_response=0.05, min_spatial_score=0.50, unwrap_aliasing=True)

    def _create_synthetic_pair(self, shift_x: int, shift_y: int, dtype=np.uint8, shape=(400, 400), val=200):
        """Helper generating synthetic reference and shifted current image."""
        h, w = shape
        ref = np.zeros((h, w), dtype=dtype)

        ref_x = 250 if shift_x < 0 else 150
        ref_y = 250 if shift_y < 0 else 150

        cv2.circle(ref, (ref_x, ref_y), 30, val, -1)
        cv2.rectangle(ref, (ref_x - 30, ref_y - 30), (ref_x - 10, ref_y - 10), val // 2, -1)

        curr = np.zeros((h, w), dtype=dtype)
        curr_x = ref_x + shift_x
        curr_y = ref_y + shift_y
        cv2.circle(curr, (curr_x, curr_y), 30, val, -1)
        cv2.rectangle(curr, (curr_x - 30, curr_y - 30), (curr_x - 10, curr_y - 10), val // 2, -1)
        return ref, curr

    def test_identical_static_frames(self):
        ref, _ = self._create_synthetic_pair(shift_x=0, shift_y=0)
        curr = ref.copy()
        res = self.engine.register(ref, curr)

        self.assertTrue(res.valid, "Identical static frames must be valid.")
        self.assertAlmostEqual(res.dx, 0.0, delta=0.5)
        self.assertAlmostEqual(res.dy, 0.0, delta=0.5)
        self.assertAlmostEqual(res.spatial_score, 1.0, delta=0.05)
        self.assertGreaterEqual(res.quality_score, 0.90)

    def test_clean_small_translation_reliable(self):
        ref, curr = self._create_synthetic_pair(shift_x=15, shift_y=10)
        res = self.engine.register(ref, curr)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, 15.0, delta=0.5)
        self.assertAlmostEqual(res.dy, 10.0, delta=0.5)
        self.assertGreater(res.spatial_score, 0.80)

    def test_clean_large_translation_reliable(self):
        ref, curr = self._create_synthetic_pair(shift_x=120, shift_y=80, shape=(400, 400))
        res = self.engine.register(ref, curr)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, 120.0, delta=1.0)
        self.assertAlmostEqual(res.dy, 80.0, delta=1.0)
        self.assertGreater(res.spatial_score, 0.70)

    def test_exact_aliasing_overlap_40_percent_reliable(self):
        # Shift -240 px on 400 px width (40% overlap)
        ref, curr = self._create_synthetic_pair(shift_x=-240, shift_y=-60, shape=(400, 400))
        res = self.engine.register(ref, curr)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, -240.0, delta=1.0)
        self.assertAlmostEqual(res.dy, -60.0, delta=1.0)
        self.assertGreater(res.spatial_score, 0.50)

    def test_weak_no_overlap_unreliable(self):
        # Image pair with no overlapping features (random noise vs zero)
        ref = np.random.randint(0, 255, size=(200, 200), dtype=np.uint8)
        curr = np.zeros((200, 200), dtype=np.uint8)

        res = self.engine.register(ref, curr)

        self.assertFalse(res.valid, "Zero overlap / blank frame must be marked invalid.")

    def test_uint16_input_reliable(self):
        ref, curr = self._create_synthetic_pair(shift_x=-50, shift_y=-30, dtype=np.uint16, shape=(400, 400), val=5000)
        res = self.engine.register(ref, curr)

        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.dx, -50.0, delta=0.5)
        self.assertAlmostEqual(res.dy, -30.0, delta=0.5)
        self.assertGreater(res.spatial_score, 0.80)

    def test_invalid_mismatched_inputs(self):
        ref = np.zeros((100, 100), dtype=np.uint8)
        curr_mismatched = np.zeros((120, 100), dtype=np.uint8)

        res_none = self.engine.register(None, ref)
        self.assertFalse(res_none.valid)

        res_mismatch = self.engine.register(ref, curr_mismatched)
        self.assertFalse(res_mismatch.valid)

    def test_input_arrays_unmodified(self):
        ref, curr = self._create_synthetic_pair(shift_x=15, shift_y=10, dtype=np.uint16)
        ref_copy = ref.copy()
        curr_copy = curr.copy()

        self.engine.register(ref, curr)

        np.testing.assert_array_equal(ref, ref_copy)
        np.testing.assert_array_equal(curr, curr_copy)


if __name__ == "__main__":
    unittest.main()
