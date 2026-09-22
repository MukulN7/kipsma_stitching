"""Real offline 1x5 mosaic integration test using the MIST Phase Image Tiles dataset.

Pipeline under test:
    SampleImageSequenceSource
    → RegistrationEngine
    → MosaicManager
    → MosaicCanvasRenderer

Target:
    Row 1, columns 1–5 (img_Phase_r001_c001.tif … img_Phase_r001_c005.tif)

Saves diagnostic mosaic rendering to outputs/mist_row1_no_blending.tif.
"""

import os
import unittest
import numpy as np
import tifffile

DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "datasets",
    "Phase_Image_Tiles",
    "Phase_Image_Tiles",
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "outputs",
)


def _dataset_available() -> bool:
    """Return True only when the MIST dataset directory exists and is populated."""
    return os.path.isdir(DATASET_DIR) and len(
        [f for f in os.listdir(DATASET_DIR) if f.endswith(".tif")]
    ) >= 5


@unittest.skipUnless(_dataset_available(), "MIST Phase Image Tiles dataset not found")
class TestRealMosaicIntegration(unittest.TestCase):
    """End-to-end integration test creating a 1x5 tile mosaic from real MIST row-1 tiles."""

    TILE_COUNT = 5  # r001_c001 … r001_c005

    def setUp(self):
        from microstitch.frame_source import SampleImageSequenceSource
        from microstitch.registration import RegistrationEngine
        from microstitch.mosaic import MosaicManager
        from microstitch.mosaic_renderer import MosaicCanvasRenderer

        self.source = SampleImageSequenceSource(DATASET_DIR)
        self.engine = RegistrationEngine()
        self.manager = MosaicManager()
        self.renderer = MosaicCanvasRenderer()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def tearDown(self):
        self.source.release()

    def test_real_1x5_mosaic_pipeline(self):
        """Build and render a 1x5 mosaic from 5 real MIST row-1 tiles."""
        frames = []
        for i in range(self.TILE_COUNT):
            ok, frame = self.source.read()
            self.assertTrue(ok, f"Failed to read frame {i + 1}/{self.TILE_COUNT}")
            self.assertIsNotNone(frame)
            frames.append(frame)

        # 1. Add origin frame (c001)
        origin_tile = self.manager.add_first_frame(frames[0])
        self.assertIsNotNone(origin_tile)
        self.assertTrue(origin_tile.is_origin)

        accepted_count = 1
        rejected_count = 0

        # 2. Sequentially register and add subsequent tiles
        prev_frame = frames[0]
        for i in range(1, self.TILE_COUNT):
            curr_frame = frames[i]
            reg_result = self.engine.register(prev_frame, curr_frame)

            if reg_result.valid:
                tile = self.manager.add_frame(curr_frame, reg_result)
                self.assertIsNotNone(tile)
                accepted_count += 1
                prev_frame = curr_frame
            else:
                rejected_count += 1

        # 3. Verify acceptance counts and mosaic properties
        self.assertEqual(accepted_count, self.TILE_COUNT, "All 5 tiles must be accepted into mosaic")
        self.assertEqual(rejected_count, 0, "No row-1 MIST tiles should be rejected")
        self.assertEqual(self.manager.tile_count, self.TILE_COUNT)

        # 4. Verify global bounds
        min_x, min_y, max_x, max_y = self.manager.get_global_bounds()
        self.assertNotEqual((min_x, min_y, max_x, max_y), (0.0, 0.0, 0.0, 0.0))
        self.assertGreater(max_x - min_x, 5000.0, "Global X span must exceed 5000px for 5 tiles")

        # 5. Render unblended and blended canvases
        canvas_unblended = self.renderer.render(self.manager, blend=False)
        canvas_blended = self.renderer.render(self.manager, blend=True)

        self.assertIsNotNone(canvas_blended)
        self.assertGreater(canvas_blended.size, 0)
        self.assertEqual(canvas_blended.dtype, np.uint16, "Blended mosaic canvas dtype must be uint16")
        self.assertEqual(canvas_blended.shape, canvas_unblended.shape, "Blended and unblended canvas dimensions must match")

        tile_h = frames[0].shape[0]
        self.assertGreater(canvas_blended.shape[1], 5000, "Canvas width must be substantially wider than one tile")
        self.assertLessEqual(canvas_blended.shape[0], tile_h + 40, "Canvas height must account for small vertical drift")

        # 6. Save diagnostic output images
        unblended_path = os.path.join(OUTPUT_DIR, "mist_row1_no_blending.tif")
        blended_path = os.path.join(OUTPUT_DIR, "mist_row1_blended.tif")

        tifffile.imwrite(unblended_path, canvas_unblended)
        tifffile.imwrite(blended_path, canvas_blended)

        self.assertTrue(os.path.exists(unblended_path), "Unblended output image file must exist")
        self.assertTrue(os.path.exists(blended_path), "Blended output image file must exist")

        # 7. Print diagnostic report to stdout (-s flag visibility)
        print("\n\n" + "=" * 65)
        print("REAL 1x5 MIST MOSAIC INTEGRATION TEST REPORT")
        print("=" * 65)
        print(f"Accepted Tiles: {accepted_count} / {self.TILE_COUNT}")
        print(f"Rejected Tiles: {rejected_count}")
        print("-" * 65)
        print(f"{'Tile Index':<12} {'Global X (px)':>15} {'Global Y (px)':>15} {'Is Origin':>12}")
        print("-" * 65)
        for tile in self.manager.get_tiles():
            print(f"{tile.frame_index:<12} {tile.global_x:>15.2f} {tile.global_y:>15.2f} {str(tile.is_origin):>12}")
        print("-" * 65)
        print(f"Global Bounds (min_x, min_y, max_x, max_y):")
        print(f"  ({min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f})")
        print(f"Final Canvas Shape: {canvas_blended.shape}")
        print(f"Final Canvas Dtype: {canvas_blended.dtype}")
        print(f"Unblended Output Saved: {unblended_path}")
        print(f"Blended Output Saved:   {blended_path}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    unittest.main()
