from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from microstitch.registration import RegistrationResult


@dataclass
class MosaicTile:
    """Metadata representation of an accepted frame positioned within the global mosaic.

    Attributes:
        frame_index (int): 0-based sequential index of the tile in the mosaic.
        global_x (float): Accumulated global X coordinate (pixels) of the top-left tile origin.
        global_y (float): Accumulated global Y coordinate (pixels) of the top-left tile origin.
        shape (Tuple[int, int]): Tile dimensions (height, width).
        dtype (np.dtype): Frame pixel data type.
        quality_score (float): Quality score [0.0, 1.0] from registration.
        response (float): Phase correlation peak response strength.
        is_origin (bool): True if this tile is the initial reference origin (0, 0).
        frame (Optional[np.ndarray]): Raw pixel frame array reference.
    """

    frame_index: int
    global_x: float
    global_y: float
    shape: Tuple[int, int]
    dtype: np.dtype
    quality_score: float
    response: float
    is_origin: bool
    frame: Optional[np.ndarray] = None


class MosaicManager:
    """Global Mosaic Manager tracking tile placement in global coordinates.

    Accumulates frame-to-frame displacement vectors (dx, dy) from RegistrationEngine
    into global tile origin positions (X, Y), upholding the physical coordinate convention:
    - Positive dx (features in curr_frame moved RIGHT relative to ref_frame) -> global_x decreases by dx.
    - Positive dy (features in curr_frame moved DOWN relative to ref_frame) -> global_y decreases by dy.
    """

    def __init__(self):
        self._tiles: List[MosaicTile] = []
        self._current_x: float = 0.0
        self._current_y: float = 0.0

    def reset(self) -> None:
        """Clears all placed tiles and resets global coordinates to origin (0, 0)."""
        self._tiles.clear()
        self._current_x = 0.0
        self._current_y = 0.0

    def add_first_frame(self, frame: np.ndarray) -> Optional[MosaicTile]:
        """Positions the initial frame at the global origin (0, 0).

        Args:
            frame (np.ndarray): The initial reference image frame.

        Returns:
            Optional[MosaicTile]: The created MosaicTile at origin, or None if frame is invalid.
        """
        if frame is None or frame.size == 0:
            return None

        self.reset()
        tile = MosaicTile(
            frame_index=0,
            global_x=0.0,
            global_y=0.0,
            shape=frame.shape[:2],
            dtype=frame.dtype,
            quality_score=1.0,
            response=1.0,
            is_origin=True,
            frame=frame,
        )
        self._tiles.append(tile)
        return tile

    def add_frame(self, frame: np.ndarray, reg_result: RegistrationResult) -> Optional[MosaicTile]:
        """Positions a new frame in global coordinates based on a valid registration result.

        Args:
            frame (np.ndarray): Incoming image frame.
            reg_result (RegistrationResult): Result from RegistrationEngine.

        Returns:
            Optional[MosaicTile]: The created MosaicTile if registration is valid, or None if rejected.
        """
        if frame is None or frame.size == 0 or reg_result is None:
            return None

        # Reject invalid registration results
        if not reg_result.valid:
            return None

        # Automatically handle initial frame if mosaic is empty
        if not self._tiles:
            return self.add_first_frame(frame)

        # Accumulate global coordinates using coordinate convention:
        # X_new = X_prev - dx, Y_new = Y_prev - dy
        self._current_x -= float(reg_result.dx)
        self._current_y -= float(reg_result.dy)

        tile = MosaicTile(
            frame_index=len(self._tiles),
            global_x=self._current_x,
            global_y=self._current_y,
            shape=frame.shape[:2],
            dtype=frame.dtype,
            quality_score=reg_result.quality_score,
            response=reg_result.response,
            is_origin=False,
            frame=frame,
        )
        self._tiles.append(tile)
        return tile

    def get_tiles(self) -> List[MosaicTile]:
        """Returns a list of all accepted mosaic tiles."""
        return list(self._tiles)

    def get_global_bounds(self) -> Tuple[float, float, float, float]:
        """Calculates global bounding box across all placed tiles.

        Returns:
            Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y).
            Returns (0.0, 0.0, 0.0, 0.0) if no tiles are placed.
        """
        if not self._tiles:
            return 0.0, 0.0, 0.0, 0.0

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for tile in self._tiles:
            h, w = tile.shape
            x1 = tile.global_x
            y1 = tile.global_y
            x2 = x1 + w
            y2 = y1 + h

            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        return min_x, min_y, max_x, max_y

    @property
    def tile_count(self) -> int:
        """Returns the number of accepted tiles currently in the mosaic."""
        return len(self._tiles)

    @property
    def current_position(self) -> Tuple[float, float]:
        """Returns current global position (X, Y)."""
        return self._current_x, self._current_y
