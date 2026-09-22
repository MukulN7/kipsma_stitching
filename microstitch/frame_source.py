from abc import ABC, abstractmethod
import glob
import os
import re
from typing import List, Optional, Tuple
import cv2
import numpy as np
import tifffile


class BaseFrameSource(ABC):
    """Abstract base class for all frame sources in MicroStitch."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame).
            success is True if a valid frame was returned, False otherwise.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases underlying camera hardware or file sequence handles."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Returns True if the frame source is opened and ready to yield frames."""
        pass


class SampleImageSequenceSource(BaseFrameSource):
    """Frame source streaming sequential images from a directory of TIFF tiles.

    Defaults to sorting by row and column grid indices (e.g. r001_c001 ... r010_c010).
    """

    def __init__(self, directory_path: str, pattern: str = "*.tif"):
        self.directory_path = os.path.abspath(directory_path)
        self.pattern = pattern
        self._image_paths: List[str] = []
        self._index: int = 0
        self._is_opened: bool = False
        self._open_sequence()

    def _extract_grid_key(self, filepath: str) -> Tuple[int, int]:
        """Extracts (row, col) grid indices from filenames like 'img_Phase_r001_c001.tif'."""
        basename = os.path.basename(filepath)
        match = re.search(r"r(\d+)_c(\d+)", basename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0

    def _open_sequence(self) -> None:
        if not os.path.exists(self.directory_path):
            self._is_opened = False
            return

        search_path = os.path.join(self.directory_path, self.pattern)
        found_files = glob.glob(search_path)

        if not found_files:
            search_path = os.path.join(self.directory_path, "**", self.pattern)
            found_files = glob.glob(search_path, recursive=True)

        if not found_files:
            self._is_opened = False
            return

        self._image_paths = sorted(found_files, key=self._extract_grid_key)
        self._index = 0
        self._is_opened = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened or self._index >= len(self._image_paths):
            return False, None

        image_path = self._image_paths[self._index]
        self._index += 1

        try:
            # Read using tifffile to preserve exact 16-bit uint16 dtype without auto-scaling
            frame = tifffile.imread(image_path)
            return True, frame
        except Exception:
            # Fallback to OpenCV IMREAD_UNCHANGED if tifffile encounters an error
            frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                return False, None
            return True, frame

    def release(self) -> None:
        self._is_opened = False
        self._index = len(self._image_paths)

    def is_opened(self) -> bool:
        return self._is_opened and self._index < len(self._image_paths)

    @property
    def total_frames(self) -> int:
        return len(self._image_paths)

    @property
    def current_index(self) -> int:
        return self._index


class USBCameraSource(BaseFrameSource):
    """Frame source for live USB camera capture using OpenCV VideoCapture."""

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ):
        """
        Args:
            camera_index (int): Device index of the USB camera (default: 0).
            width (Optional[int]): Requested frame width in pixels.
            height (Optional[int]): Requested frame height in pixels.
            fps (Optional[float]): Requested frame rate (FPS).
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        try:
            self._cap = cv2.VideoCapture(self.camera_index)
            if self._cap is not None and self._cap.isOpened():
                if self.width is not None:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                if self.height is not None:
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                if self.fps is not None:
                    self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))
                # Set buffer size to 1 frame to prevent streaming latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._is_opened = True
            else:
                self.release()
        except Exception:
            self.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened() or self._cap is None:
            return False, None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame
        except Exception:
            return False, None

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        return bool(self._is_opened and self._cap is not None and self._cap.isOpened())
