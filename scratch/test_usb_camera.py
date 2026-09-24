import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from microstitch.frame_source import USBCameraSource


def main():
    parser = argparse.ArgumentParser(description="Test USB Camera live frame capture")
    parser.add_argument("camera_index", type=int, nargs="?", default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Optional frame width")
    parser.add_argument("--height", type=int, default=None, help="Optional frame height")
    args = parser.parse_args()

    camera = USBCameraSource(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height
    )

    is_opened = camera.is_opened()
    print(f"Camera index: {args.camera_index}")
    print(f"Camera opened: {is_opened}")

    if not is_opened:
        print(f"Error: Unable to open camera at index {args.camera_index}", file=sys.stderr)
        camera.release()
        sys.exit(1)

    success, frame = camera.read()
    if not success or frame is None:
        print(f"Error: Failed to read frame from camera at index {args.camera_index}", file=sys.stderr)
        camera.release()
        sys.exit(1)

    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Min: {np.min(frame)}")
    print(f"Max: {np.max(frame)}")
    print(f"Mean: {np.mean(frame):.2f}")

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "camera_test_frame.png"

    saved = cv2.imwrite(str(output_path), frame)
    if not saved:
        print(f"Error: Failed to save frame to {output_path}", file=sys.stderr)
        camera.release()
        sys.exit(1)

    print(f"Saved captured frame to: {output_path}")

    camera.release()
    print("Camera released cleanly.")


if __name__ == "__main__":
    main()
