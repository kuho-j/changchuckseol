"""Serve the Raspberry Pi camera as an MJPEG stream.

Run this on the Raspberry Pi over SSH, then open the displayed URL in a web
browser.  For an SSH-only network, use port forwarding, for example:

    ssh -L 8000:localhost:8000 pi@<raspberry-pi-host>

and open http://localhost:8000 on the local computer.
"""

from __future__ import annotations

import argparse
import signal
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Final

import cv2

from src.capture import VideoCapture


BOUNDARY: Final = "frame"


class CameraStream:
    """Convert frames from ``VideoCapture`` to JPEG safely for web clients."""

    def __init__(self, camera_num: int, size: tuple[int, int], quality: int) -> None:
        self._camera = VideoCapture(camera_num=camera_num, size=size)
        self._quality = quality
        self._lock = Lock()

    def start(self) -> None:
        self._camera.start()

    def jpeg_frame(self) -> bytes:
        """Return the newest RGB camera frame encoded as JPEG."""
        with self._lock:
            rgb_frame = self._camera.capture_frame()

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if not ok:
            raise RuntimeError("카메라 프레임을 JPEG로 인코딩하지 못했습니다.")
        return encoded.tobytes()

    def close(self) -> None:
        self._camera.close()


def make_handler(stream: CameraStream) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to one shared camera stream."""

    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path in {"/", "/index.html"}:
                self._serve_page()
            elif self.path == "/stream.mjpg":
                self._serve_stream()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_page(self) -> None:
            page = f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><title>Camera stream</title>
<style>body{{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}}
img{{max-width:100vw;max-height:100vh;display:block;margin:auto}}</style></head>
<body><img src=\"/stream.mjpg\" alt=\"실시간 카메라 화면\"></body></html>""".encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def _serve_stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()

            try:
                while True:
                    frame = stream.jpeg_frame()
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser closed or reloaded the page.

        def log_message(self, format: str, *args: object) -> None:
            # HTTPS/TLS bytes sent to this HTTP-only server can otherwise be
            # displayed as garbled terminal characters in a 400 error log.
            message = format % args
            safe_message = message.encode("unicode_escape").decode("ascii")
            print(f"{self.client_address[0]} - {safe_message}")

    return StreamHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi camera MJPEG server")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="frame height (default: 480)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)")
    args = parser.parse_args()
    if args.port not in range(1, 65536):
        parser.error("--port must be between 1 and 65535")
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if args.quality not in range(1, 101):
        parser.error("--quality must be between 1 and 100")
    return args


def main() -> None:
    args = parse_args()
    stream = CameraStream(args.camera, (args.width, args.height), args.quality)
    stream.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(stream))
    server.daemon_threads = True

    print(f"카메라 스트림: http://localhost:{args.port}")
    print("종료하려면 Ctrl+C를 누르세요.")

    def shutdown(*_: object) -> None:
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with suppress(Exception):
            stream.close()


if __name__ == "__main__":
    main()
