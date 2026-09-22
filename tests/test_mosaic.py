import unittest
import numpy as np
from microstitch.mosaic import MosaicManager, MosaicTile
from microstitch.registration import RegistrationResult


class TestMosaicManager(unittest.TestCase):
    def setUp(self):
        self.manager = MosaicManager()

    def test_first_frame_becomes_origin(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        tile = self.manager.add_first_frame(frame)

        self.assertIsNotNone(tile)
        self.assertEqual(tile.frame_index, 0)
        self.assertEqual(tile.global_x, 0.0)
        self.assertEqual(tile.global_y, 0.0)
        self.assertEqual(tile.shape, (100, 200))
        self.assertTrue(tile.is_origin)
        self.assertEqual(self.manager.tile_count, 1)

    def test_one_positive_x_translation(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame1)

        reg = RegistrationResult(dx=20.0, dy=0.0, response=0.8, valid=True, spatial_score=0.9, quality_score=0.9)
        tile2 = self.manager.add_frame(frame2, reg)

        self.assertIsNotNone(tile2)
        self.assertEqual(tile2.global_x, -20.0)
        self.assertEqual(tile2.global_y, 0.0)
        self.assertFalse(tile2.is_origin)

    def test_one_negative_x_translation(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame1)

        reg = RegistrationResult(dx=-25.0, dy=0.0, response=0.8, valid=True, spatial_score=0.9, quality_score=0.9)
        tile2 = self.manager.add_frame(frame2, reg)

        self.assertIsNotNone(tile2)
        self.assertEqual(tile2.global_x, 25.0)
        self.assertEqual(tile2.global_y, 0.0)

    def test_one_positive_y_translation(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame1)

        reg = RegistrationResult(dx=0.0, dy=15.0, response=0.8, valid=True, spatial_score=0.9, quality_score=0.9)
        tile2 = self.manager.add_frame(frame2, reg)

        self.assertIsNotNone(tile2)
        self.assertEqual(tile2.global_x, 0.0)
        self.assertEqual(tile2.global_y, -15.0)

    def test_one_negative_y_translation(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame1)

        reg = RegistrationResult(dx=0.0, dy=-18.0, response=0.8, valid=True, spatial_score=0.9, quality_score=0.9)
        tile2 = self.manager.add_frame(frame2, reg)

        self.assertIsNotNone(tile2)
        self.assertEqual(tile2.global_x, 0.0)
        self.assertEqual(tile2.global_y, 18.0)

    def test_combined_xy_translation(self):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame1)

        reg = RegistrationResult(dx=30.0, dy=-12.0, response=0.85, valid=True, spatial_score=0.95, quality_score=0.95)
        tile2 = self.manager.add_frame(frame2, reg)

        self.assertIsNotNone(tile2)
        self.assertEqual(tile2.global_x, -30.0)
        self.assertEqual(tile2.global_y, 12.0)

    def test_multiple_sequential_frames_accumulation(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame)  # (0, 0)

        step1 = RegistrationResult(dx=10.0, dy=5.0, response=0.8, valid=True)
        step2 = RegistrationResult(dx=10.0, dy=5.0, response=0.8, valid=True)
        step3 = RegistrationResult(dx=-5.0, dy=15.0, response=0.8, valid=True)

        t1 = self.manager.add_frame(frame, step1)  # (-10, -5)
        t2 = self.manager.add_frame(frame, step2)  # (-20, -10)
        t3 = self.manager.add_frame(frame, step3)  # (-15, -25)

        self.assertEqual(self.manager.tile_count, 4)
        self.assertEqual(t1.global_x, -10.0)
        self.assertEqual(t1.global_y, -5.0)
        self.assertEqual(t2.global_x, -20.0)
        self.assertEqual(t2.global_y, -10.0)
        self.assertEqual(t3.global_x, -15.0)
        self.assertEqual(t3.global_y, -25.0)

    def test_rejected_invalid_registration(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame)

        reg_invalid = RegistrationResult(dx=50.0, dy=50.0, response=0.01, valid=False)
        tile = self.manager.add_frame(frame, reg_invalid)

        self.assertIsNone(tile, "Invalid registration must be rejected.")
        self.assertEqual(self.manager.tile_count, 1)
        self.assertEqual(self.manager.current_position, (0.0, 0.0))

    def test_uint16_frame_metadata(self):
        frame = np.zeros((1040, 1392), dtype=np.uint16)
        tile = self.manager.add_first_frame(frame)

        self.assertEqual(tile.dtype, np.uint16)
        self.assertEqual(tile.shape, (1040, 1392))

    def test_input_frame_unmodified(self):
        frame = np.random.randint(1000, 5000, size=(100, 100), dtype=np.uint16)
        frame_copy = frame.copy()

        self.manager.add_first_frame(frame)
        reg = RegistrationResult(dx=10.0, dy=10.0, response=0.9, valid=True)
        self.manager.add_frame(frame, reg)

        np.testing.assert_array_equal(frame, frame_copy)

    def test_reset_behavior(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        self.manager.add_first_frame(frame)
        reg = RegistrationResult(dx=20.0, dy=20.0, response=0.8, valid=True)
        self.manager.add_frame(frame, reg)

        self.assertEqual(self.manager.tile_count, 2)
        self.manager.reset()

        self.assertEqual(self.manager.tile_count, 0)
        self.assertEqual(self.manager.current_position, (0.0, 0.0))
        self.assertEqual(self.manager.get_global_bounds(), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
