from typing import Optional, Tuple
import numpy as np


class MovementDetector:
    """Detects meaningful spatial or structural movement between consecutive frames.

    Uses Mean Absolute Difference (MAD) calculated non-destructively on float32 copies
    of incoming frames to support both 8-bit uint8 and 16-bit uint16 grayscale frames.
    """

    def __init__(self, threshold: float = 5.0, auto_update_reference: bool = True):
        """
        Args:
            threshold (float): Mean Absolute Difference (MAD) threshold. Frames with MAD >= threshold
                               are classified as having meaningful movement.
            auto_update_reference (bool): If True, automatically updates the reference frame
                                          when meaningful movement is detected.
        """
        self.threshold: float = threshold
        self.auto_update_reference: bool = auto_update_reference
        self._reference_frame: Optional[np.ndarray] = None
        self._last_mad: float = 0.0

    def reset(self) -> None:
        """Resets the reference frame state."""
        self._reference_frame = None
        self._last_mad = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Processes an incoming frame and determines if meaningful movement occurred.

        Args:
            frame (np.ndarray): Input image frame (2D grayscale or 3D color).

        Returns:
            Tuple[bool, float]: (moved, mad_score).
                - moved: True if movement detected (or if initial frame), False otherwise.
                - mad_score: The calculated Mean Absolute Difference score.
        """
        if frame is None:
            return False, 0.0

        # Create float32 copy to prevent mutating source input frame
        working_frame = self._to_grayscale_float32(frame)

        if self._reference_frame is None:
            self._reference_frame = working_frame
            self._last_mad = 0.0
            return True, 0.0

        # Calculate Mean Absolute Difference
        diff = np.abs(working_frame - self._reference_frame)
        mad = float(np.mean(diff))
        self._last_mad = mad

        has_moved = mad >= self.threshold

        if has_moved and self.auto_update_reference:
            self._reference_frame = working_frame

        return has_moved, mad

    def has_moved(self, frame: np.ndarray) -> bool:
        """Convenience method returning True if frame contains movement relative to reference."""
        moved, _ = self.process_frame(frame)
        return moved

    @staticmethod
    def _to_grayscale_float32(frame: np.ndarray) -> np.ndarray:
        """Converts frame to 2D float32 without modifying the original input array."""
        if frame.ndim == 3:
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            return np.dot(frame[..., :3], weights).astype(np.float32)
        else:
            return frame.astype(np.float32, copy=True)

    @property
    def last_mad(self) -> float:
        return self._last_mad

    @property
    def is_initialized(self) -> bool:
        return self._reference_frame is not None
