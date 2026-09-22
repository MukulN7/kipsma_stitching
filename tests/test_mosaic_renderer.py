import unittest
import numpy as np
from microstitch.mosaic import MosaicManager
from microstitch.mosaic_renderer import MosaicCanvasRenderer
from microstitch.registration import RegistrationResult


class TestMosaicCanvasRenderer(unittest.TestCase):
    def setUp(self):
        self.manager = MosaicManager()
        self.renderer = MosaicCanvasRenderer()

    def test_empty_mosaic_behavior(self):
        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.size, 0)
        self.assertEqual(canvas.dtype, np.uint8)

    def test_one_tile_at_origin(self):
        frame = np.full((100, 100), fill_value=150, dtype=np.uint8)
        self.manager.add_first_frame(frame)

        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.shape, (100, 100))
        self.assertEqual(canvas.dtype, np.uint8)
        np.testing.assert_array_equal(canvas, frame)

    def test_two_tiles_positive_displacement(self):
        # Frame 1 at (0, 0), Frame 2 with dx=-20, dy=-30 -> Global pos (+20, +30)
        frame1 = np.full((100, 100), fill_value=100, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=200, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        reg = RegistrationResult(dx=-20.0, dy=-30.0, response=0.9, valid=True)
        self.manager.add_frame(frame2, reg)

        canvas = self.renderer.render(self.manager)
        # Expected canvas dimensions: Height = 30 + 100 = 130, Width = 20 + 100 = 120
        self.assertEqual(canvas.shape, (130, 120))
        # Non-overlapping tile 1 region
        self.assertEqual(canvas[0, 0], 100)
        # Non-overlapping tile 2 region
        self.assertEqual(canvas[129, 119], 200)

    def test_two_tiles_negative_displacement(self):
        # Frame 1 at (0, 0), Frame 2 with dx=20, dy=30 -> Global pos (-20, -30)
        frame1 = np.full((100, 100), fill_value=100, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=200, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        reg = RegistrationResult(dx=20.0, dy=30.0, response=0.9, valid=True)
        self.manager.add_frame(frame2, reg)

        canvas = self.renderer.render(self.manager, blend=False)
        # Bounds: min_x = -20, max_x = 100 -> Width = 120
        # Bounds: min_y = -30, max_y = 100 -> Height = 130
        self.assertEqual(canvas.shape, (130, 120))
        # Frame 2 top-left is translated to canvas (0, 0)
        self.assertEqual(canvas[0, 0], 200)
        # Frame 1 top-left is translated to canvas (30, 20)
        self.assertEqual(canvas[30, 20], 200)  # Overwritten by frame 2 if overlap exists

    def test_combined_xy_displacement(self):
        frame1 = np.full((100, 100), fill_value=50, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=150, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        reg = RegistrationResult(dx=15.0, dy=-25.0, response=0.9, valid=True)
        self.manager.add_frame(frame2, reg)

        canvas = self.renderer.render(self.manager)
        # Global pos frame 2: (-15, +25)
        # Bounds: min_x = -15, max_x = 100 -> Width = 115
        # Bounds: min_y = 0, max_y = 125 -> Height = 125
        self.assertEqual(canvas.shape, (125, 115))

    def test_multiple_tiles_grid(self):
        frame = np.full((50, 50), fill_value=100, dtype=np.uint8)
        self.manager.add_first_frame(frame)  # (0, 0)

        # Move right: dx=-50 -> (50, 0)
        self.manager.add_frame(frame, RegistrationResult(dx=-50.0, dy=0.0, response=0.9, valid=True))
        # Move down: dy=-50 -> (50, 50)
        self.manager.add_frame(frame, RegistrationResult(dx=0.0, dy=-50.0, response=0.9, valid=True))
        # Move left: dx=50 -> (0, 50)
        self.manager.add_frame(frame, RegistrationResult(dx=50.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.shape, (100, 100))

    def test_overlapping_tiles_overwrite_behavior(self):
        # Frame 1 filled with 100, Frame 2 filled with 200, placed at 50% overlap (dx=-50, dy=0)
        frame1 = np.full((100, 100), fill_value=100, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=200, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        self.manager.add_frame(frame2, RegistrationResult(dx=-50.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager, blend=False)
        # Canvas size: (100, 150)
        self.assertEqual(canvas.shape, (100, 150))
        # Non-overlapping frame 1 region (x=0 to 49)
        self.assertEqual(canvas[50, 20], 100)
        # Overlapping region (x=50 to 99) must be overwritten by frame 2 (value 200) when blend=False
        self.assertEqual(canvas[50, 75], 200)
        # Frame 2 region (x=100 to 149)
        self.assertEqual(canvas[50, 120], 200)

    def test_overlapping_tiles_linear_feather_blend(self):
        # Frame 1 filled with 100, Frame 2 filled with 200, placed at 50% overlap (dx=-50, dy=0)
        frame1 = np.full((100, 100), fill_value=100, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=200, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        self.manager.add_frame(frame2, RegistrationResult(dx=-50.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager, blend=True)
        self.assertEqual(canvas.shape, (100, 150))
        # Non-overlapping frame 1 region
        self.assertEqual(canvas[50, 20], 100)
        # Non-overlapping frame 2 region
        self.assertEqual(canvas[50, 120], 200)
        # Center of overlap (x=74, x=75): equal weights -> 150
        self.assertAlmostEqual(int(canvas[50, 74]), 150, delta=2)
        # Smooth transition: pixel near left edge of overlap (x=50) should be closer to 100
        self.assertLess(int(canvas[50, 51]), 110)
        # Smooth transition: pixel near right edge of overlap (x=99) should be closer to 200
        self.assertGreater(int(canvas[50, 98]), 190)

    def test_overlapping_identical_tiles_blend(self):
        # Identical overlapping tiles must yield the exact intensity in the overlap region
        frame1 = np.full((100, 100), fill_value=120, dtype=np.uint8)
        frame2 = np.full((100, 100), fill_value=120, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        self.manager.add_frame(frame2, RegistrationResult(dx=-30.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager, blend=True)
        np.testing.assert_array_equal(canvas, 120)

    def test_uint16_blending(self):
        frame1 = np.full((50, 50), fill_value=10000, dtype=np.uint16)
        frame2 = np.full((50, 50), fill_value=20000, dtype=np.uint16)

        self.manager.add_first_frame(frame1)
        self.manager.add_frame(frame2, RegistrationResult(dx=-25.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager, blend=True)
        self.assertEqual(canvas.dtype, np.uint16)
        self.assertEqual(canvas[25, 10], 10000)
        self.assertEqual(canvas[25, 60], 20000)
        self.assertAlmostEqual(canvas[25, 37], 15000, delta=100)

    def test_rgb_color_blending(self):
        frame1 = np.full((40, 40, 3), fill_value=50, dtype=np.uint8)
        frame2 = np.full((40, 40, 3), fill_value=150, dtype=np.uint8)

        self.manager.add_first_frame(frame1)
        self.manager.add_frame(frame2, RegistrationResult(dx=-20.0, dy=0.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager, blend=True)
        self.assertEqual(canvas.ndim, 3)
        self.assertEqual(canvas.shape, (40, 60, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_uint8_dtype_preservation(self):
        frame = np.full((50, 50), fill_value=255, dtype=np.uint8)
        self.manager.add_first_frame(frame)
        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.dtype, np.uint8)

    def test_uint16_dtype_preservation(self):
        frame = np.full((50, 50), fill_value=6000, dtype=np.uint16)
        self.manager.add_first_frame(frame)
        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.dtype, np.uint16)
        self.assertEqual(canvas[25, 25], 6000)

    def test_correct_handling_negative_coordinates(self):
        frame = np.full((50, 50), fill_value=123, dtype=np.uint8)
        self.manager.add_first_frame(frame)
        # Shift to negative global coordinates: dx=40, dy=30 -> (-40, -30)
        self.manager.add_frame(frame, RegistrationResult(dx=40.0, dy=30.0, response=0.9, valid=True))

        canvas = self.renderer.render(self.manager)
        # Canvas bounds: min_x=-40, max_x=50 -> 90; min_y=-30, max_y=50 -> 80
        self.assertEqual(canvas.shape, (80, 90))
        self.assertEqual(canvas[0, 0], 123)  # Top-left of shifted tile at canvas (0, 0)


if __name__ == "__main__":
    unittest.main()
