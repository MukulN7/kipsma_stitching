import os
import unittest
import numpy as np
from microstitch.frame_source import BaseFrameSource, SampleImageSequenceSource, USBCameraSource


class TestFrameSource(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = os.path.abspath("datasets/Phase_Image_Tiles/Phase_Image_Tiles")

    def test_sample_image_sequence_source_dataset_open(self):
        source = SampleImageSequenceSource(self.dataset_dir)
        self.assertTrue(source.is_opened(), "SampleImageSequenceSource failed to open valid dataset directory.")
        self.assertEqual(source.total_frames, 100, f"Expected 100 tiles in MIST dataset, found {source.total_frames}")

    def test_sample_image_sequence_source_first_frame(self):
        source = SampleImageSequenceSource(self.dataset_dir)
        success, frame = source.read()
        self.assertTrue(success, "Failed to read first frame from dataset.")
        self.assertIsNotNone(frame, "First frame returned None.")
        self.assertEqual(frame.shape, (1040, 1392), f"Expected shape (1040, 1392), got {frame.shape}")
        self.assertEqual(frame.dtype, np.uint16, f"Expected dtype uint16, got {frame.dtype}")

    def test_sample_image_sequence_source_multiple_frames(self):
        source = SampleImageSequenceSource(self.dataset_dir)
        for i in range(5):
            success, frame = source.read()
            self.assertTrue(success, f"Failed to read frame at index {i}")
            self.assertEqual(frame.shape, (1040, 1392))
            self.assertEqual(frame.dtype, np.uint16)
        self.assertEqual(source.current_index, 5)

    def test_sample_image_sequence_source_end_of_sequence(self):
        source = SampleImageSequenceSource(self.dataset_dir)
        count = 0
        while True:
            success, frame = source.read()
            if not success:
                break
            count += 1
            self.assertIsNotNone(frame)
        
        self.assertEqual(count, 100, f"Expected 100 frames read before end-of-sequence, got {count}")
        # Verify extra reads after end of sequence return False cleanly
        extra_success, extra_frame = source.read()
        self.assertFalse(extra_success)
        self.assertIsNone(extra_frame)
        self.assertFalse(source.is_opened())

    def test_sample_image_sequence_source_invalid_dir(self):
        source = SampleImageSequenceSource("non_existent_directory_12345")
        self.assertFalse(source.is_opened())
        success, frame = source.read()
        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_usb_camera_source_disconnected_handling(self):
        # Use invalid camera index 999 to test disconnected error handling
        camera = USBCameraSource(camera_index=999)
        self.assertFalse(camera.is_opened())
        success, frame = camera.read()
        self.assertFalse(success)
        self.assertIsNone(frame)
        camera.release()
        self.assertFalse(camera.is_opened())

    def test_usb_camera_source_mocked_success(self):
        from unittest.mock import MagicMock, patch

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc:
            camera = USBCameraSource(camera_index=0, width=640, height=480, fps=30.0)
            mock_vc.assert_called_once_with(0)
            self.assertTrue(camera.is_opened())

            # Verify camera property settings
            import cv2
            mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
            mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
            mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30.0)
            mock_cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Test frame read
            success, frame = camera.read()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)
            np.testing.assert_array_equal(frame, fake_frame)

            # Test release
            camera.release()
            mock_cap.release.assert_called_once()
            self.assertFalse(camera.is_opened())

    def test_usb_camera_source_mocked_failed_open(self):
        from unittest.mock import MagicMock, patch

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            camera = USBCameraSource(camera_index=1)
            self.assertFalse(camera.is_opened())

            success, frame = camera.read()
            self.assertFalse(success)
            self.assertIsNone(frame)

    def test_usb_camera_source_mocked_failed_read(self):
        from unittest.mock import MagicMock, patch

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            camera = USBCameraSource(camera_index=0)
            self.assertTrue(camera.is_opened())

            success, frame = camera.read()
            self.assertFalse(success)
            self.assertIsNone(frame)

    def test_interface_polymorphism(self):
        seq_source = SampleImageSequenceSource(self.dataset_dir)
        cam_source = USBCameraSource(camera_index=999)

        self.assertIsInstance(seq_source, BaseFrameSource)
        self.assertIsInstance(cam_source, BaseFrameSource)


if __name__ == "__main__":
    unittest.main()
