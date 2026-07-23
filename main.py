import cv2
import time

from src.camera import Camera
from src.face_mesh import FaceMeshDetector
from src.config import *

def main():

    camera = Camera()

    mesh = FaceMeshDetector()

    previous_time = time.time()

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = mesh.process(frame)

        frame = mesh.draw(frame, results)

        current_time = time.time()

        fps = 1 / (current_time - previous_time)

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS : {int(fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            GREEN,
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            break

    camera.release()

if __name__ == "__main__":
    main()