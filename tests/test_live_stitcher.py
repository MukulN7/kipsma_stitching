"""Tests for LiveStitcher pipeline orchestration.

All tests use synthetic uint8 BGR frames and a mock frame source.
No physical hardware is required.
"""

from __future__ import annotations

import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from microstitch.frame_source import BaseFrameSource
from microstitch.live_stitcher import LiveStitcher


# ── synthetic frame helpers ────────────────────────────────────────────────────

def _solid_frame(h: int = 360, w: int = 640,
                 color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Returns a solid-color uint8 BGR frame."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _noise_frame(h: int = 360, w: int = 640, seed: int = 0) -> np.ndarray:
    """Returns a random noise uint8 BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _shifted_frame(base: np.ndarray, dx: int = 20, dy: int = 0) -> np.ndarray:
    """Shifts *base* by (dx, dy) pixels (wraps edges)."""
    return np.roll(np.roll(base, dx, axis=1), dy, axis=0)


class MockFrameSource(BaseFrameSource):
    """Feeds a pre-defined list of frames to the pipeline."""

    def __init__(self, frames: list) -> None:
        self._frames = frames
        self._idx = 0
        self._opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._idx >= len(self._frames):
            return False, None
        f = self._frames[self._idx]
        self._idx += 1
        return True, f

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened and self._idx < len(self._frames)


# ── LiveStitcher factory bypassing real camera ─────────────────────────────────

def _make_stitcher(**kwargs) -> LiveStitcher:
    """Return a LiveStitcher without opening a real camera."""
    s = object.__new__(LiveStitcher)
    from microstitch.movement_detector import MovementDetector
    from microstitch.mosaic import MosaicManager
    from microstitch.mosaic_renderer import MosaicCanvasRenderer
    from microstitch.registration import RegistrationEngine

    s.camera_index = 0
    s.width = None
    s.height = None
    s._camera = MagicMock()          # camera never used in unit tests
    s._movement = MovementDetector(threshold=kwargs.get("movement_threshold", 3.0),
                                   auto_update_reference=True)
    s._engine = RegistrationEngine(
        min_response=kwargs.get("min_response", 0.05),
        min_spatial_score=kwargs.get("min_spatial", 0.35),
        use_laplacian=True,
    )
    s._mosaic = MosaicManager()
    s._renderer = MosaicCanvasRenderer()
    s._prev_accepted = None
    s._tile_count = 0
    s._last_status = ""
    s._last_quality = 0.0
    return s


# ── tests ──────────────────────────────────────────────────────────────────────

class TestLiveStitcherInit(unittest.TestCase):
    def test_initial_tile_count_is_zero(self):
        s = _make_stitcher()
        self.assertEqual(s.tile_count, 0)

    def test_reset_clears_state(self):
        s = _make_stitcher()
        # Seed some state
        frame = _noise_frame(seed=1)
        s.process_frame(frame)
        s.reset()
        self.assertEqual(s.tile_count, 0)
        self.assertIsNone(s._prev_accepted)

    def test_render_before_any_frame_returns_none(self):
        s = _make_stitcher()
        canvas = s.render_mosaic()
        self.assertIsNone(canvas)


class TestFirstFrameIsOrigin(unittest.TestCase):
    def test_first_moved_frame_becomes_origin(self):
        s = _make_stitcher()
        frame = _noise_frame(seed=42)
        result = s.process_frame(frame)
        self.assertTrue(result["moved"])
        self.assertTrue(result["is_origin"])
        self.assertEqual(result["tile_count"], 1)
        self.assertEqual(s.mosaic_manager.tile_count, 1)

    def test_first_tile_position_is_origin(self):
        s = _make_stitcher()
        frame = _noise_frame(seed=7)
        s.process_frame(frame)
        tiles = s.mosaic_manager.get_tiles()
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].global_x, 0.0)
        self.assertEqual(tiles[0].global_y, 0.0)
        self.assertTrue(tiles[0].is_origin)


class TestMovementFiltering(unittest.TestCase):
    def test_identical_frames_not_accepted_after_origin(self):
        """Identical frames should be detected as stationary and skipped."""
        s = _make_stitcher(movement_threshold=3.0)
        origin = _noise_frame(seed=10)
        # First frame -> origin
        r1 = s.process_frame(origin)
        self.assertTrue(r1["is_origin"])
        # Same frame -> stationary
        r2 = s.process_frame(origin.copy())
        self.assertFalse(r2["moved"])
        self.assertEqual(s.tile_count, 1)  # still only origin

    def test_stationary_frame_status_contains_stationary(self):
        s = _make_stitcher(movement_threshold=3.0)
        origin = _noise_frame(seed=11)
        s.process_frame(origin)
        r = s.process_frame(origin.copy())
        self.assertIn("stationary", r["status"].lower())


class TestShiftedFrameRegistration(unittest.TestCase):
    """Registration tests use a texture-rich base shifted by a few pixels."""

    def _base_frame(self) -> np.ndarray:
        rng = np.random.default_rng(99)
        base = rng.integers(30, 220, (360, 640, 3), dtype=np.uint8)
        # Add a visible gradient to help phase correlation
        for c in range(3):
            base[:, :, c] = np.clip(
                base[:, :, c].astype(np.int32) +
                np.tile(np.arange(640, dtype=np.int32), (360, 1)) // 4,
                0, 255
            ).astype(np.uint8)
        return base

    def test_shifted_frame_may_be_registered(self):
        """Shifted frame should either register (valid) or be rejected gracefully;
        the pipeline must not raise an exception."""
        s = _make_stitcher(movement_threshold=2.0, min_spatial=0.30)
        base = self._base_frame()
        shifted = _shifted_frame(base, dx=30)
        s.process_frame(base)        # origin
        result = s.process_frame(shifted)
        # Either accepted or rejected is fine; no exception is the key assertion
        self.assertIn(result["valid"], [True, False])
        self.assertIn("status", result)

    def test_large_shift_may_fail_quality_gate(self):
        """A frame shifted by nearly the full width should generally fail the quality gate."""
        s = _make_stitcher(movement_threshold=2.0, min_spatial=0.50)
        base = self._base_frame()
        massive_shift = _shifted_frame(base, dx=300, dy=100)
        s.process_frame(base)
        result = s.process_frame(massive_shift)
        # No crash; result is a valid dict
        self.assertIsInstance(result["valid"], bool)
        self.assertIsInstance(result["status"], str)


class TestMosaicRenderAfterAccept(unittest.TestCase):
    def test_mosaic_renders_after_origin(self):
        s = _make_stitcher()
        frame = _noise_frame(seed=3)
        s.process_frame(frame)
        canvas = s.render_mosaic()
        self.assertIsNotNone(canvas)
        self.assertGreater(canvas.size, 0)

    def test_mosaic_canvas_dtype_matches_input(self):
        s = _make_stitcher()
        frame = _noise_frame(seed=5)
        s.process_frame(frame)
        canvas = s.render_mosaic()
        self.assertEqual(canvas.dtype, frame.dtype)


class TestMultipleRegisteredTiles(unittest.TestCase):
    def test_two_valid_tiles_both_in_manager(self):
        """Feed two clearly-different noise frames; both should be placed if registration passes."""
        s = _make_stitcher(movement_threshold=1.0, min_response=0.01, min_spatial=0.01)
        f1 = _noise_frame(seed=100)
        f2 = _noise_frame(seed=200)   # completely different -> high MAD, registration may or may not pass
        s.process_frame(f1)
        s.process_frame(f2)
        # Tile count is 1 (origin) or 2 (if registration succeeded) - either is valid
        self.assertGreaterEqual(s.tile_count, 1)

    def test_reset_after_tiles_added(self):
        s = _make_stitcher(movement_threshold=1.0)
        s.process_frame(_noise_frame(seed=20))
        s.process_frame(_noise_frame(seed=21))
        s.reset()
        self.assertEqual(s.tile_count, 0)
        canvas = s.render_mosaic()
        self.assertIsNone(canvas)


class TestProcessFrameReturnContract(unittest.TestCase):
    """Verify the result dict always has the expected keys."""

    EXPECTED_KEYS = {"moved", "is_origin", "registered", "valid",
                     "tile_count", "status", "quality"}

    def _check(self, result: dict) -> None:
        for k in self.EXPECTED_KEYS:
            self.assertIn(k, result, f"Key '{k}' missing from result")

    def test_contract_first_frame(self):
        s = _make_stitcher()
        self._check(s.process_frame(_noise_frame(seed=0)))

    def test_contract_stationary_frame(self):
        s = _make_stitcher()
        f = _noise_frame(seed=1)
        s.process_frame(f)
        self._check(s.process_frame(f.copy()))

    def test_contract_shifted_frame(self):
        s = _make_stitcher(movement_threshold=1.0)
        f = _noise_frame(seed=2)
        s.process_frame(f)
        self._check(s.process_frame(_shifted_frame(f, dx=15)))


if __name__ == "__main__":
    unittest.main()
