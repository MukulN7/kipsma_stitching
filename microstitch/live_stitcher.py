"""Live microscope stitching pipeline.

Connects USBCameraSource -> MovementDetector -> RegistrationEngine
-> MosaicManager -> MosaicCanvasRenderer into a single interactive
OpenCV session.

Usage
-----
From the project root::

    python -m microstitch.live_stitcher            # camera 0
    python -m microstitch.live_stitcher 1          # camera 1 (USB microscope)
    python -m microstitch.live_stitcher 1 640 360  # with explicit resolution

Keys
----
  Q  - quit
  R  - reset mosaic
  S  - save current mosaic to outputs/live_mosaic.tif
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile

from microstitch.frame_source import USBCameraSource
from microstitch.mosaic import MosaicManager
from microstitch.mosaic_renderer import MosaicCanvasRenderer
from microstitch.registration import RegistrationEngine

# ÔöÇÔöÇ tunable constants ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
_MIN_RESPONSE      = 0.05    # phase-correlation peak required
_MIN_SPATIAL       = 0.35    # spatial cross-correlation required (relaxed for live cam)
_MAX_SHIFT         = 150.0   # px; maximum accepted displacement magnitude for live microscope stream
_MOSAIC_DISPLAY_W  = 640     # display width for the rendered mosaic preview


# ÔöÇÔöÇ helpers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

def _resize_for_display(img: np.ndarray, target_width: int) -> np.ndarray:
    """Scale *img* so that its width equals *target_width*, preserving aspect ratio."""
    if img is None or img.size == 0:
        return np.zeros((10, target_width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if w == 0:
        return np.zeros((10, target_width, 3), dtype=np.uint8)
    scale = target_width / w
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized


def _to_display_u8(img: np.ndarray) -> np.ndarray:
    """Convert any dtype image to uint8 BGR for display."""
    if img.ndim == 2:
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            rng = float(mx - mn) if mx != mn else 1.0
            img = ((img.astype(np.float32) - mn) / rng * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        rng = float(mx - mn) if mx != mn else 1.0
        img = ((img.astype(np.float32) - mn) / rng * 255).astype(np.uint8)
    return img


def _overlay_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw *lines* as a white-on-black HUD onto a copy of *img*."""
    out = img.copy()
    y = 20
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


def _save_mosaic(manager: MosaicManager, renderer: MosaicCanvasRenderer, out_dir: Path) -> None:
    tiles = manager.get_tiles()
    if not tiles:
        print("[Save] No tiles to save.")
        return
    canvas = renderer.render(manager, blend=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_mosaic.tif"
    tifffile.imwrite(str(out_path), canvas)
    print(f"[Save] Mosaic saved to {out_path}  ({canvas.shape}, {canvas.dtype})")


# ÔöÇÔöÇ orchestrator ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

class LiveStitcher:
    """Orchestrates continuous frame capture and mosaic building.

    Parameters
    ----------
    camera_index : int
        cv2.VideoCapture device index.
    width, height : int | None
        Optional resolution request forwarded to USBCameraSource.
    movement_threshold : float
        MAD threshold forwarded to MovementDetector.
    min_response, min_spatial : float
        Quality gate thresholds forwarded to RegistrationEngine.
    max_shift : float | None
        Maximum displacement magnitude in pixels forwarded to RegistrationEngine.
    """

    def __init__(
        self,
        camera_index: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        min_response: float = _MIN_RESPONSE,
        min_spatial: float = _MIN_SPATIAL,
        max_shift: Optional[float] = _MAX_SHIFT,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._camera = USBCameraSource(camera_index, width=width, height=height)
        self._engine = RegistrationEngine(
            min_response=min_response,
            min_spatial_score=min_spatial,
            max_shift=max_shift,
            use_laplacian=True,
            use_sift=True,
        )
        self._mosaic = MosaicManager()
        self._renderer = MosaicCanvasRenderer()

        # Pipeline state
        self._prev_frame: Optional[np.ndarray] = None  # immediately previous camera frame
        self._tile_count: int = 0
        self._last_status: str = "Waiting for first frame..."
        self._last_quality: float = 0.0

    # ÔöÇÔöÇ public API (used in tests) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def reset(self) -> None:
        """Resets mosaic and pipeline state."""
        self._mosaic.reset()
        self._prev_frame = None
        self._tile_count = 0
        self._last_status = "Reset."
        self._last_quality = 0.0

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame through the full pipeline.

        Parameters
        ----------
        frame : np.ndarray
            Raw camera/test frame.

        Returns
        -------
        dict with keys:
          - moved (bool)
          - is_origin (bool)
          - registered (bool)
          - valid (bool)
          - tile_count (int)
          - status (str)
          - quality (float)
        """
        result = {
            "moved": True,
            "is_origin": False,
            "registered": False,
            "valid": False,
            "tile_count": self._tile_count,
            "status": "processing",
            "quality": 0.0,
        }

        # ÔöÇÔöÇ first frame: mosaic origin ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        if self._prev_frame is None:
            self._mosaic.add_first_frame(frame)
            self._prev_frame = frame.copy()
            self._tile_count = 1
            result.update({
                "is_origin": True,
                "registered": True,
                "valid": True,
                "tile_count": 1,
                "status": "origin placed",
                "quality": 1.0,
            })
            self._last_status = "Origin placed."
            self._last_quality = 1.0
            return result

        # ÔöÇÔöÇ register against immediately previous frame ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        ref_frame = self._prev_frame
        self._prev_frame = frame.copy()  # ALWAYS update reference frame for next frame

        try:
            reg = self._engine.register(ref_frame, frame)
        except Exception as exc:
            result["status"] = f"registration error: {exc}"
            self._last_status = result["status"]
            return result

        result["quality"] = reg.quality_score
        self._last_quality = reg.quality_score

        if not reg.valid:
            result["status"] = (
                f"REJECTED (response={reg.response:.3f} "
                f"spatial={reg.spatial_score:.3f})"
            )
            self._last_status = result["status"]
            return result

        # ÔöÇÔöÇ valid registration: add to mosaic ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        tile = self._mosaic.add_frame(frame, reg)
        if tile is None:
            result["status"] = "tile rejected by MosaicManager"
            self._last_status = result["status"]
            return result

        self._tile_count = self._mosaic.tile_count
        result.update({
            "registered": True,
            "valid": True,
            "tile_count": self._tile_count,
            "status": (
                f"OK  dx={reg.dx:+.1f} dy={reg.dy:+.1f} "
                f"q={reg.quality_score:.2f}"
            ),
            "quality": reg.quality_score,
        })
        self._last_status = result["status"]
        return result

    def render_mosaic(self) -> Optional[np.ndarray]:
        """Renders the current mosaic canvas.  Returns None if no tiles."""
        if self._mosaic.tile_count == 0:
            return None
        return self._renderer.render(self._mosaic, blend=True)

    @property
    def tile_count(self) -> int:
        return self._tile_count

    @property
    def mosaic_manager(self) -> MosaicManager:
        return self._mosaic

    @property
    def renderer(self) -> MosaicCanvasRenderer:
        return self._renderer

    # ÔöÇÔöÇ interactive run loop ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def run(self) -> None:
        """Open the camera and run the interactive stitching session."""
        if not self._camera.is_opened():
            print(f"[Error] Cannot open camera at index {self.camera_index}.", file=sys.stderr)
            sys.exit(1)

        print(f"[Live] Camera {self.camera_index} opened.  Q=quit  R=reset  S=save")
        out_dir = Path("outputs")
        mosaic_canvas: Optional[np.ndarray] = None

        while True:
            ok, frame = self._camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Process through pipeline
            info = self.process_frame(frame)
            if info["valid"]:
                mosaic_canvas = self.render_mosaic()

            # ÔöÇÔöÇ camera preview ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
            cam_disp = _to_display_u8(frame.copy())
            cam_hud = _overlay_text(cam_disp, [
                f"Camera {self.camera_index}  [{frame.shape[1]}x{frame.shape[0]}]",
                f"Tiles: {info['tile_count']}",
                f"Status: {info['status']}",
                "Q=quit  R=reset  S=save",
            ])
            cv2.imshow("Live Camera", cam_hud)

            # ÔöÇÔöÇ mosaic preview ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
            if mosaic_canvas is not None and mosaic_canvas.size > 0:
                mos_disp = _to_display_u8(mosaic_canvas)
                mos_disp = _resize_for_display(mos_disp, _MOSAIC_DISPLAY_W)
                mos_hud = _overlay_text(mos_disp, [
                    f"Mosaic  tiles={info['tile_count']}",
                    f"q={self._last_quality:.2f}  {self._last_status}",
                ])
                cv2.imshow("Live Mosaic", mos_hud)

            # ÔöÇÔöÇ key handling ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[Live] Quit.")
                break
            elif key == ord("r"):
                self.reset()
                mosaic_canvas = None
                print("[Live] Mosaic reset.")
            elif key == ord("s"):
                _save_mosaic(self._mosaic, self._renderer, out_dir)

        self._camera.release()
        cv2.destroyAllWindows()


# ÔöÇÔöÇ entry point ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicroStitch live stitching pipeline")
    p.add_argument("camera_index", type=int, nargs="?", default=1,
                   help="Camera device index (default: 1)")
    p.add_argument("width", type=int, nargs="?", default=None,
                   help="Optional frame width")
    p.add_argument("height", type=int, nargs="?", default=None,
                   help="Optional frame height")
    p.add_argument("--min-response", type=float, default=_MIN_RESPONSE,
                   help=f"Phase-correlation min response (default: {_MIN_RESPONSE})")
    p.add_argument("--min-spatial", type=float, default=_MIN_SPATIAL,
                   help=f"Spatial cross-correlation min score (default: {_MIN_SPATIAL})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stitcher = LiveStitcher(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        min_response=args.min_response,
        min_spatial=args.min_spatial,
    )
    stitcher.run()
