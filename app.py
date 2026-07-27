#!/usr/bin/env python3
"""
Emoji Rain AI - application entry point.

Running ``python app.py`` opens the default webcam, detects the user's
face in real time using MediaPipe Face Mesh, classifies the current facial
expression using a geometry-driven rule engine, and renders a continuous
"rain" of emojis that match the detected expression, on top of a modern
glassmorphism HUD.

Controls:
    q / ESC   Quit the application.
    l         Toggle raw landmark dot overlay (debug aid).
    h         Toggle the HUD stats panel.
    d         Toggle the debug features panel.

See README.md for full installation and usage instructions.
"""

from __future__ import annotations

import sys

import cv2
from src.app_state import AppState, CameraStatus, DetectionStatus
from src.camera import Camera, CameraError
from src.config import DEFAULT_CONFIG
from src.emoji_engine import EmojiEngine
from src.expression_analyzer import ExpressionAnalyzer
from src.landmark_detector import LandmarkDetector
from src.logger import get_logger
from src.particle_system import ParticleSystem
from src.renderer import Renderer
from src.ui import UIOverlay
from src.utils import DeltaTimer, FPSCounter

logger = get_logger("app")


class EmojiRainApp:
    """Wires together every subsystem and owns the main render loop.

    This class is intentionally a thin orchestrator: all real logic lives
    in the single-responsibility modules under ``src/`` (camera capture,
    landmark detection, expression analysis, particle simulation,
    rendering). ``EmojiRainApp`` just calls them in the right order each
    frame and manages the OpenCV window lifecycle.
    """

    def __init__(self) -> None:
        self.config = DEFAULT_CONFIG
        self.state = AppState()

        self.camera = Camera(self.config.camera)
        self.landmark_detector = LandmarkDetector(self.config.landmarks)
        self.expression_analyzer = ExpressionAnalyzer(self.config.expression)
        self.emoji_engine = EmojiEngine(self.config.emoji)
        self.particle_system = ParticleSystem(self.config.particles)
        self.renderer = Renderer(self.config.particles, self.config.ui)
        self.ui_overlay = UIOverlay(self.config.ui, self.renderer)

        self.fps_counter = FPSCounter(window_size=30)
        self.delta_timer = DeltaTimer()

        self._show_hud = True
        self._show_landmarks = False
        self._show_debug_panel = True  # Default to True for visibility
        self._running = False

    def run(self) -> int:
        """Starts the camera and runs the main loop until the user quits
        or an unrecoverable error occurs.

        Returns:
            A process exit code (``0`` on clean shutdown, non-zero on
            fatal error).
        """
        try:
            self.camera.open()
        except CameraError as exc:
            logger.error("Fatal: could not start camera: %s", exc)
            self.state.camera_status = CameraStatus.ERROR
            return 1

        self.state.camera_status = CameraStatus.CONNECTED
        window_name = self.config.ui.window_title
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._running = True

        logger.info("Emoji Rain AI started. Press 'q' or ESC to quit.")

        try:
            while self._running:
                self._run_single_frame(window_name)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        except Exception:  # noqa: BLE001 - top-level safety net
            logger.exception("Unhandled exception in main loop.")
            return 1
        finally:
            self._shutdown()

        return 0

    def _run_single_frame(self, window_name: str) -> None:
        success, frame = self.camera.read()
        if not success or frame is None:
            logger.warning("Failed to read frame from camera.")
            self.state.camera_status = CameraStatus.DISCONNECTED
            self._running = False
            return

        self.state.camera_status = CameraStatus.CONNECTED
        frame_height, frame_width = frame.shape[:2]
        delta_time = self.delta_timer.tick()

        detection = self.landmark_detector.process(frame)

        if detection.detected and detection.landmarks is not None:
            self.state.detection_status = DetectionStatus.FACE_DETECTED
            self.state.landmark_count = detection.landmark_count

            result = self.expression_analyzer.analyze(
                detection.landmarks, frame, frame_width, frame_height
            )
            self.state.current_expression = result.expression
            self.state.expression_confidence = result.confidence
            # Store debug data
            self.state.debug_features = result.features
            self.state.debug_raw_candidate = result.raw_candidate

            style = self.emoji_engine.update(result.expression)
            self.particle_system.set_spawn_enabled(True)
        else:
            self.state.detection_status = DetectionStatus.FACE_LOST
            self.state.landmark_count = 0
            self.state.debug_features = None
            self.state.debug_raw_candidate = None
            # No face: stop spawning new particles, but let existing ones
            # finish their natural fade-out/float-away animation.
            self.particle_system.set_spawn_enabled(False)
            style = self.emoji_engine.current_style

        self.particle_system.update(delta_time, style, frame_width, frame_height)
        self.state.particle_count = self.particle_system.count
        self.state.total_particles_spawned = self.particle_system.total_spawned

        canvas = self.renderer.begin_frame(frame)
        self.renderer.draw_particles(canvas, self.particle_system.particles)
        if self._show_hud:
            self.ui_overlay.draw(canvas, self.state, style)
        if self._show_debug_panel:
            self.ui_overlay.draw_debug_panel(canvas, self.state)
        output_frame = self.renderer.finish_frame(canvas)

        current_fps = self.fps_counter.tick()
        self.state.record_frame(current_fps)

        cv2.imshow(window_name, output_frame)
        self._handle_keys()

    def _handle_keys(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            logger.info("Quit requested by user.")
            self._running = False
        elif key == ord("h"):
            self._show_hud = not self._show_hud
            logger.debug("HUD visibility toggled: %s", self._show_hud)
        elif key == ord("l"):
            self._show_landmarks = not self._show_landmarks
            logger.debug("Landmark overlay toggled: %s", self._show_landmarks)
        elif key == ord("d"):
            self._show_debug_panel = not self._show_debug_panel
            logger.debug("Debug panel toggled: %s", self._show_debug_panel)

    def _shutdown(self) -> None:
        logger.info(
            "Shutting down. Frames processed: %d, particles spawned: %d.",
            self.state.frame_count,
            self.particle_system.total_spawned,
        )
        self.camera.release()
        self.landmark_detector.close()
        cv2.destroyAllWindows()


def main() -> int:
    """Console-script / ``python app.py`` entry point."""
    app = EmojiRainApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
