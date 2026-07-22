"""
Main Application Entry Point

Live webcam feed with FPS counter.
Press 'Q' to quit.
"""

import cv2
import sys

from src.camera import Camera
from src.config import WINDOW_NAME, QUIT_KEY


def main():
    """
    Main application loop.
    """
    camera = None

    try:
        print("📷 Starting webcam...")
        camera = Camera()
        print(f"✅ Webcam opened successfully! Resolution: {camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Press 'Q' to quit")

        while True:
            success, frame, fps = camera.read()

            if not success:
                print("⚠️ Failed to capture frame.")
                break

            # Add FPS overlay
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show the frame
            cv2.imshow(WINDOW_NAME, frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                print("🛑 Closing application...")
                break

    except Exception as error:
        print(f"❌ Error: {error}")
        sys.exit(1)

    finally:
        if camera is not None:
            camera.release()
            print("✅ Camera released successfully.")


if __name__ == "__main__":
    main()