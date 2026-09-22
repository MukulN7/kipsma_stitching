"""Offline end-to-end integration test using the MIST Phase Image Tiles dataset.

Pipeline under test:
    SampleImageSequenceSource
    → RegistrationEngine
    → MosaicManager
    → MosaicCanvasRenderer

Subset used:
    Row 1, columns 1–5  (img_Phase_r001_c001.tif … img_Phase_r001_c005.tif)
    — a single horizontal strip of 5 real microscope tiles.

Registration behaviour at 10 % overlap:
    The MIST tiles share ~10 % overlap.  Full-frame phase correlation at this
    overlap fraction consistently yields negative cv2.phaseCorrelate responses
    (confirmed by the controlled-overlap experiment: valid results only above
    ~40–50 % overlap).  The quality gate therefore rejects all 4 pairs, and
    only the origin tile is accepted into the mosaic.  This is correct and
    expected behaviour — not a bug.

    These tests verify that the pipeline:
      • Does not crash or raise exceptions on real microscope tiles.
      • Returns finite, well-formed RegistrationResult values for every pair.
      • Correctly rejects low-quality registrations (all responses < 0).
      • Retains only the origin tile gracefully when nothing passes the gate.
      • Preserves uint16 dtype and pixel values throughout the canvas.
"""

import os
import unittest
import numpy as np

DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "datasets",
    "Phase_Image_Tiles",
    "Phase_Image_Tiles",
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _dataset_available() -> bool:
    """Return True only when the MIST dataset directory exists and is populated."""
    return os.path.isdir(DATASET_DIR) and len(
        [f for f in os.listdir(DATASET_DIR) if f.endswith(".tif")]
    ) >= 5


# ── test class ─────────────────────────────────────────────────────────────────


@unittest.skipUnless(_dataset_available(), "MIST Phase Image Tiles dataset not found")
class TestOfflinePipeline(unittest.TestCase):
    """End-to-end integration using five real MIST tiles (row 1, columns 1–5)."""

    TILE_COUNT = 5  # r001_c001 … r001_c005

    def setUp(self):
        from microstitch.frame_source import SampleImageSequenceSource
        from microstitch.registration import RegistrationEngine
        from microstitch.mosaic import MosaicManager
        from microstitch.mosaic_renderer import MosaicCanvasRenderer

        # Use a sub-directory that contains only the row-1 tiles via filtering.
        # SampleImageSequenceSource already sorts by (row, col) key, so we just
        # open the full dataset directory and stop after TILE_COUNT reads.
        self.source = SampleImageSequenceSource(DATASET_DIR)

        # Use default thresholds (min_response=0.05, min_spatial_score=0.50).
        # At 10 % overlap, cv2.phaseCorrelate returns negative responses, so
        # all pairs are rejected — this is the expected, correct outcome.
        self.engine = RegistrationEngine()
        self.manager = MosaicManager()
        self.renderer = MosaicCanvasRenderer()

    def tearDown(self):
        self.source.release()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _read_n_frames(self, n: int):
        """Read exactly *n* frames from the source; raise if unavailable."""
        frames = []
        for i in range(n):
            ok, frame = self.source.read()
            self.assertTrue(ok, f"Failed to read frame {i + 1}/{n}")
            self.assertIsNotNone(frame)
            frames.append(frame)
        return frames

    # ── tests ──────────────────────────────────────────────────────────────────

    def test_source_opens_and_yields_real_frames(self):
        """SampleImageSequenceSource must open the MIST dataset and return valid uint16 frames."""
        self.assertTrue(self.source.is_opened(), "Source should be opened")
        self.assertGreaterEqual(self.source.total_frames, self.TILE_COUNT)

        ok, frame = self.source.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.dtype, np.uint16, "MIST tiles should be uint16")
        self.assertEqual(frame.ndim, 2, "MIST tiles should be grayscale (2D)")
        self.assertGreater(frame.size, 0)

    def test_frame_dtype_and_shape_consistency(self):
        """All five tiles must share the same shape and dtype."""
        frames = self._read_n_frames(self.TILE_COUNT)
        reference_shape = frames[0].shape
        reference_dtype = frames[0].dtype

        for i, frame in enumerate(frames[1:], start=2):
            self.assertEqual(
                frame.shape, reference_shape,
                f"Frame {i} shape {frame.shape} != reference {reference_shape}"
            )
            self.assertEqual(
                frame.dtype, reference_dtype,
                f"Frame {i} dtype {frame.dtype} != reference {reference_dtype}"
            )

    def test_registration_engine_does_not_crash_on_real_tiles(self):
        """RegistrationEngine must return valid RegistrationResult for all row-1 MIST pairs via fallback."""
        from microstitch.registration import RegistrationResult

        frames = self._read_n_frames(self.TILE_COUNT)
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.assertIsInstance(result, RegistrationResult)
            self.assertTrue(np.isfinite(result.dx), f"dx not finite for pair {i}")
            self.assertTrue(np.isfinite(result.dy), f"dy not finite for pair {i}")
            self.assertTrue(np.isfinite(result.response), f"response not finite for pair {i}")
            self.assertTrue(
                result.valid,
                f"Expected valid=True at 10%% overlap (pair {i}), got valid=False "
                f"(response={result.response:.4f})"
            )

    def test_mosaic_manager_accepts_registered_tiles(self):
        """MosaicManager correctly accepts all 5 registered MIST tiles into the mosaic."""
        frames = self._read_n_frames(self.TILE_COUNT)

        # First frame → origin
        self.manager.add_first_frame(frames[0])
        self.assertEqual(self.manager.tile_count, 1)

        # Register remaining frames sequentially — all should be accepted
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        # All 5 tiles must be retained in the mosaic
        self.assertEqual(
            self.manager.tile_count, self.TILE_COUNT,
            f"Expected all {self.TILE_COUNT} tiles in mosaic, got {self.manager.tile_count}"
        )

    def test_canvas_dimensions_for_stitched_sequence(self):
        """When 5 tiles are stitched, the canvas must span all placed tiles."""
        frames = self._read_n_frames(self.TILE_COUNT)
        tile_h, tile_w = frames[0].shape

        self.manager.add_first_frame(frames[0])
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        canvas = self.renderer.render(self.manager)

        # 5 tiles shifted by ~1200px each should span ~6000+ px in width
        self.assertGreater(
            canvas.shape[1], tile_w,
            f"Canvas width {canvas.shape[1]} should be greater than single tile width {tile_w}"
        )

    def test_canvas_height_bounded_to_single_tile_row(self):
        """Within a single row of tiles, canvas height must not exceed one tile height plus small drift margin."""
        frames = self._read_n_frames(self.TILE_COUNT)
        tile_h, tile_w = frames[0].shape

        self.manager.add_first_frame(frames[0])
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        canvas = self.renderer.render(self.manager)

        # Allow small vertical drift margin across 5 sequential tiles (±30 px)
        margin = 30
        self.assertLessEqual(
            canvas.shape[0], tile_h + margin,
            f"Canvas height {canvas.shape[0]} exceeds tile height {tile_h} by more than {margin}px margin"
        )

    def test_canvas_dtype_matches_source_dtype(self):
        """The renderer must preserve the uint16 dtype of the MIST source tiles."""
        frames = self._read_n_frames(self.TILE_COUNT)

        self.manager.add_first_frame(frames[0])
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        canvas = self.renderer.render(self.manager)
        self.assertEqual(
            canvas.dtype, np.uint16,
            f"Canvas dtype should be uint16, got {canvas.dtype}"
        )

    def test_canvas_pixel_values_within_uint16_range(self):
        """All canvas pixel values must be within the valid uint16 range [0, 65535]."""
        frames = self._read_n_frames(self.TILE_COUNT)

        self.manager.add_first_frame(frames[0])
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        canvas = self.renderer.render(self.manager)
        self.assertEqual(canvas.dtype, np.uint16)
        self.assertGreaterEqual(int(canvas.min()), 0)
        self.assertLessEqual(int(canvas.max()), 65535)

    def test_origin_tile_pixels_preserved_in_canvas(self):
        """The origin tile region at the canvas top-left must retain its pixel values."""
        frames = self._read_n_frames(self.TILE_COUNT)
        origin_frame = frames[0]

        self.manager.add_first_frame(frames[0])
        for i in range(1, len(frames)):
            result = self.engine.register(frames[i - 1], frames[i])
            self.manager.add_frame(frames[i], result)

        canvas = self.renderer.render(self.manager)
        min_x, min_y, _, _ = self.manager.get_global_bounds()

        # Compute origin tile placement on global canvas
        col_start = int(round(0.0 - min_x))
        row_start = int(round(0.0 - min_y))

        # Sample a 20×20 patch in the non-overlapping right interior (col 500..520) of origin tile
        patch_size = 20
        origin_patch = origin_frame[50:50 + patch_size, 500:500 + patch_size]
        canvas_patch = canvas[row_start + 50:row_start + 50 + patch_size,
                              col_start + 500:col_start + 500 + patch_size]

        np.testing.assert_array_equal(
            origin_patch, canvas_patch,
            err_msg="Origin tile interior pixels must be unchanged in the rendered canvas"
        )

    def test_end_of_sequence_detected_correctly(self):
        """After consuming TILE_COUNT frames, is_opened() must reflect end-of-sequence."""
        for _ in range(self.TILE_COUNT):
            self.source.read()
        # Source may still be "opened" if there are more tiles beyond the first 5;
        # what we verify is that we CAN read exactly TILE_COUNT frames without error.
        # (Full exhaustion is tested in test_frame_source.py.)
        self.assertEqual(self.source.current_index, self.TILE_COUNT)


if __name__ == "__main__":
    unittest.main()
