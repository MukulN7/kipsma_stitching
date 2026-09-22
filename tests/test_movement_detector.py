import unittest
import numpy as np
from microstitch.movement_detector import MovementDetector


class TestMovementDetector(unittest.TestCase):
    def test_first_frame_establishes_reference(self):
        detector = MovementDetector(threshold=5.0)
        self.assertFalse(detector.is_initialized)

        frame = np.ones((100, 100), dtype=np.uint8) * 100
        moved, mad = detector.process_frame(frame)

        self.assertTrue(moved, "Initial frame must return True to establish reference.")
        self.assertEqual(mad, 0.0)
        self.assertTrue(detector.is_initialized)

    def test_identical_frames_classified_as_static(self):
        detector = MovementDetector(threshold=5.0)
        frame1 = np.ones((100, 100), dtype=np.uint8) * 100
        frame2 = np.ones((100, 100), dtype=np.uint8) * 100

        detector.process_frame(frame1)
        moved, mad = detector.process_frame(frame2)

        self.assertFalse(moved, "Identical frames must be classified as static (False).")
        self.assertEqual(mad, 0.0)

    def test_shifted_different_frames_classified_as_movement(self):
        detector = MovementDetector(threshold=5.0)
        
        # Create a synthetic frame with a bright rectangle on dark background
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame1[20:50, 20:50] = 200

        # Shift the rectangle in frame2
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[40:70, 40:70] = 200

        detector.process_frame(frame1)
        moved, mad = detector.process_frame(frame2)

        self.assertTrue(moved, "Shifted synthetic frame must be classified as movement.")
        self.assertGreater(mad, 5.0)

    def test_threshold_configuration(self):
        # With low threshold 2.0, a slight difference should trigger movement
        detector_sensitive = MovementDetector(threshold=2.0)
        # With high threshold 50.0, a slight difference should NOT trigger movement
        detector_strict = MovementDetector(threshold=50.0)

        frame1 = np.ones((100, 100), dtype=np.float32) * 100.0
        frame2 = np.ones((100, 100), dtype=np.float32) * 105.0  # MAD = 5.0

        detector_sensitive.process_frame(frame1)
        moved_sens, mad_sens = detector_sensitive.process_frame(frame2)

        detector_strict.process_frame(frame1)
        moved_strict, mad_strict = detector_strict.process_frame(frame2)

        self.assertTrue(moved_sens, "Sensitive detector (thresh 2.0) should detect movement for MAD=5.0")
        self.assertFalse(moved_strict, "Strict detector (thresh 50.0) should ignore movement for MAD=5.0")
        self.assertEqual(mad_sens, 5.0)
        self.assertEqual(mad_strict, 5.0)

    def test_uint16_mist_style_frames(self):
        detector = MovementDetector(threshold=10.0)

        # Create uint16 synthetic frame mimicking MIST tile intensities (~2500)
        frame1 = np.full((1040, 1392), fill_value=2500, dtype=np.uint16)
        frame2 = np.full((1040, 1392), fill_value=2500, dtype=np.uint16)
        # Add a region of displacement
        frame2[100:300, 100:300] = 5000

        detector.process_frame(frame1)
        moved, mad = detector.process_frame(frame2)

        self.assertTrue(moved, "uint16 frame with local displacement must trigger movement.")
        self.assertGreater(mad, 10.0)

    def test_input_frame_immutability(self):
        detector = MovementDetector(threshold=5.0)

        original_data = np.random.randint(1000, 5000, size=(200, 200), dtype=np.uint16)
        frame_copy = original_data.copy()

        detector.process_frame(original_data)

        # Assert input numpy array is unchanged
        np.testing.assert_array_equal(original_data, frame_copy)
        self.assertEqual(original_data.dtype, np.uint16)
        self.assertEqual(original_data.shape, (200, 200))


if __name__ == "__main__":
    unittest.main()
