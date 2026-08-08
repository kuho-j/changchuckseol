"""Raspberry Pi camera capture utilities.

Captured frames are returned as RGB ``numpy.ndarray`` values so they can be
passed directly to preprocessing or classification code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

ImageProcessor = Callable[[np.ndarray], Any]


def _create_picamera(camera_num: int) -> Any:
    """Create Picamera2 lazily so this module can be imported off the Pi."""
    try:
        from picamera2 import Picamera2
    except ImportError as error:
        raise RuntimeError(
            "Picamera2를 찾을 수 없습니다. Raspberry Pi OS에서 실행하거나 "
            "Picamera2를 설치하세요."
        ) from error

    return Picamera2(camera_num)


class StillCapture:
    """Capture still images from a Raspberry Pi camera."""

    def __init__(
        self,
        camera_num: int = 0,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.camera_num = camera_num
        self.size = size
        self._camera: Any | None = None

    def start(self) -> None:
        """Configure the camera for still capture and start it."""
        if self._camera is not None:
            return

        camera = _create_picamera(self.camera_num)
        main_config: dict[str, Any] = {"format": "RGB888"}
        if self.size is not None:
            main_config["size"] = self.size

        camera.configure(camera.create_still_configuration(main=main_config))
        camera.start()
        self._camera = camera

    def capture(self) -> np.ndarray:
        """Capture one still image as an RGB array."""
        self.start()
        assert self._camera is not None
        return self._camera.capture_array("main")

    def close(self) -> None:
        """Stop the camera and release resources."""
        if self._camera is None:
            return

        self._camera.stop()
        self._camera.close()
        self._camera = None

    def __enter__(self) -> StillCapture:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VideoCapture:
    """Capture frames from a Raspberry Pi camera video stream."""

    def __init__(
        self,
        camera_num: int = 0,
        size: tuple[int, int] = (640, 480),
    ) -> None:
        self.camera_num = camera_num
        self.size = size
        self._camera: Any | None = None
        self._last_frame: np.ndarray | None = None

    def start(self) -> None:
        """Configure the camera for video capture and start it."""
        if self._camera is not None:
            return

        camera = _create_picamera(self.camera_num)
        camera.configure(
            camera.create_video_configuration(
                main={"format": "RGB888", "size": self.size}
            )
        )
        camera.start()
        self._camera = camera

    def capture_frame(self) -> np.ndarray:
        """Capture the next video frame and remember it as the last frame."""
        self.start()
        assert self._camera is not None
        frame = self._camera.capture_array("main")
        self._last_frame = frame
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield video frames until the caller stops iterating."""
        while True:
            yield self.capture_frame()

    def get_last_frame(self, copy: bool = True) -> np.ndarray | None:
        """Return the most recently captured video frame, if any."""
        if self._last_frame is None:
            return None
        if copy:
            return self._last_frame.copy()
        return self._last_frame

    def close(self) -> None:
        """Stop the camera and release resources."""
        if self._camera is None:
            return

        self._camera.stop()
        self._camera.close()
        self._camera = None

    def __enter__(self) -> VideoCapture:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def capture_still(
    camera_num: int = 0,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Capture one still image and return it as an RGB array."""
    with StillCapture(camera_num=camera_num, size=size) as camera:
        return camera.capture()


def capture_video_frame(
    camera_num: int = 0,
    size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Capture one frame from the video stream and return it as an RGB array."""
    with VideoCapture(camera_num=camera_num, size=size) as camera:
        return camera.capture_frame()


def save_image(
    image: np.ndarray,
    path: str | PathLike[str],
    create_parent: bool = True,
) -> Path:
    """Save an RGB image array to disk and return the saved path."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python을 설치해야 이미지를 저장할 수 있습니다.") from error

    save_path = Path(path)
    if create_parent:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(save_path), bgr_image):
        raise RuntimeError(f"이미지를 저장하지 못했습니다: {save_path}")

    return save_path


def capture_still_to_file(
    path: str | PathLike[str],
    camera_num: int = 0,
    size: tuple[int, int] | None = None,
) -> Path:
    """Capture one still image, save it to ``path``, and return the saved path."""
    image = capture_still(camera_num=camera_num, size=size)
    return save_image(image, path)


def capture_still_and_process(
    processor: ImageProcessor,
    camera_num: int = 0,
    size: tuple[int, int] | None = None,
) -> Any:
    """Capture one still image and pass it to ``processor``."""
    return processor(capture_still(camera_num=camera_num, size=size))
