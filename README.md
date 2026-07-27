# 😊 Emoji Rain AI

**Real-time facial expression detection that rains matching emojis on screen — powered by MediaPipe Face Mesh geometry, not a black-box classifier.**

[![CI](https://img.shields.io/github/actions/workflow/status/your-org/Emoji-Rain-AI/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Folder Structure](#folder-structure)
- [Emoji Font Setup](#emoji-font-setup)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Description

**Emoji Rain AI** watches your webcam, tracks your face using **468 MediaPipe Face Mesh landmarks**, and computes real facial geometry — eye aspect ratio, mouth curvature, eyebrow height, mouth opening, and more — to figure out what expression you're making. It then animates a continuous, physically-simulated **particle rain** of the matching emoji floating up from the bottom of the screen, rendered on a modern dark **glassmorphism** HUD.

There is no pretrained emotion classifier here. Every decision is driven by an explicit, interpretable, and fully tunable geometric rule engine — so you can see exactly *why* the app thinks you're smiling, and adjust the thresholds yourself.

```
python app.py
```

...opens your webcam and starts the show. Smile, wink, laugh, gasp, or frown — the emoji rain follows you.

## Features

- 🎯 **468-point facial landmark tracking** via MediaPipe Face Mesh (with iris refinement)
- 🧠 **Geometry-driven expression engine** — Eye Aspect Ratio, Mouth Aspect Ratio, smile intensity, eyebrow height, jaw opening, cheek raise, best-effort teeth/tongue visibility, and head pose (yaw/pitch/roll) via `solvePnP`
- 🌊 **Temporal smoothing** with a rolling confidence-vote history so expressions transition smoothly instead of flickering frame-to-frame
- ✨ **Custom particle engine** — every emoji is an independent particle with position, velocity, rotation, scale, opacity, drift, and lifetime, updated and faded every frame
- 🎨 **Glassmorphism dark-theme HUD** showing FPS, current expression, confidence, particle count, landmark count, detection status, and camera status
- ⚙️ **Fully configurable** — every threshold, color, spawn rate, and particle behavior lives in one `config.py`
- 🪵 **Structured logging** to console (colorized) and rotating log files
- ✅ **Unit-tested** core logic (camera, expression classification, particle simulation) with `pytest`
- 🛠️ **CI-ready** — GitHub Actions workflow runs linting, formatting checks, type-checking, and tests on every push

## Architecture

Emoji Rain AI follows clean architecture: each module owns exactly one responsibility, and dependencies flow in a single direction — from raw I/O, through analysis, to presentation.

```
┌─────────────┐     ┌───────────────────┐     ┌───────────────────────┐
│   Camera    │ --> │  LandmarkDetector  │ --> │  ExpressionAnalyzer   │
│ (webcam I/O)│     │ (MediaPipe wrapper)│     │ (geometry + smoothing)│
└─────────────┘     └───────────────────┘     └───────────┬───────────┘
                                                            │
                                                            v
┌─────────────┐     ┌───────────────────┐     ┌───────────────────────┐
│  UIOverlay  │ <-- │      Renderer      │ <-- │      EmojiEngine      │
│   (HUD)     │     │ (PIL compositing)  │     │ (expression -> glyph) │
└─────────────┘     └─────────┬─────────┘     └───────────┬───────────┘
                               │                           │
                               v                           v
                     ┌───────────────────────────────────────┐
                     │            ParticleSystem              │
                     │  (spawns / updates / prunes Particles) │
                     └─────────────────────────────────────────┘
```

`app.py` (`EmojiRainApp`) is a thin orchestrator: it owns the OpenCV window and main loop, and calls each module in the correct order every frame. All real logic lives in `src/`, where each file has a single, well-documented responsibility (`camera.py`, `landmark_detector.py`, `expression_analyzer.py`, `emoji_engine.py`, `particle.py`, `particle_system.py`, `renderer.py`, `ui.py`).

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Emoji-Rain-AI.git
cd Emoji-Rain-AI

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python **3.9+**
- A webcam
- OS packages for OpenCV's GUI backend (Linux: `libgl1`, `libglib2.0-0` — see `.github/workflows/ci.yml` for the exact apt packages used in CI)
- See [`requirements.txt`](requirements.txt) for pinned Python dependencies (`opencv-python`, `mediapipe`, `numpy`, `Pillow`)

## Usage

```bash
python app.py
```

**Controls:**

| Key       | Action                        |
|-----------|--------------------------------|
| `q` / Esc | Quit the application            |
| `h`       | Toggle the HUD stats panel      |
| `l`       | Toggle raw landmark dot overlay |

Smile 😊, grin with teeth 😁, laugh 😂, wink 😉, wink with your tongue out 😜, stick your tongue out 😛, look shocked 😲, frown 😢, or furrow your brow 😠 — the currently detected expression's emoji continuously rains upward. Change your expression and the old emoji stops spawning while the new one begins, with in-flight particles gracefully finishing their own fade-out.

## Configuration

Every tunable value lives in [`src/config.py`](src/config.py), grouped into typed dataclasses:

| Config class            | Controls |
|--------------------------|----------|
| `CameraConfig`            | Resolution, FPS, device index, mirror mode, reconnect behavior |
| `LandmarkConfig`          | MediaPipe Face Mesh detection/tracking confidence, iris refinement |
| `ExpressionThresholds`    | Every geometric threshold (EAR, MAR, smile, eyebrows) and smoothing parameters |
| `EmojiMappingConfig`      | Expression → emoji glyph and accent color mapping |
| `ParticleConfig`          | Max particles, spawn rate, speed, scale, drift, lifetime, fade behavior |
| `UIConfig`                | HUD colors, fonts, panel size, corner radius |
| `LoggingConfig`           | Console/file log levels, log rotation |

Change a value, save, and re-run — no other code changes needed.

## Folder Structure

```
Emoji-Rain-AI/
├── .github/workflows/ci.yml     # Lint, format, type-check, test on every push
├── assets/                      # Fonts, emoji assets, sounds, icons (see Emoji Font Setup)
├── docs/screenshots/            # README screenshots / demo media
├── models/                      # Optional local model cache (MediaPipe downloads automatically)
├── src/
│   ├── app_state.py             # Shared mutable per-frame state (AppState)
│   ├── camera.py                # Webcam capture wrapper (Camera)
│   ├── config.py                # All tunable configuration (AppConfig and friends)
│   ├── emoji_engine.py          # Expression -> emoji glyph/color mapping (EmojiEngine)
│   ├── expression_analyzer.py   # Rule-based classifier + temporal smoothing (ExpressionAnalyzer)
│   ├── geometry.py              # Pure landmark geometry functions (EAR, MAR, smile, head pose...)
│   ├── landmark_detector.py     # MediaPipe Face Mesh wrapper (LandmarkDetector)
│   ├── logger.py                # Application-wide logging setup
│   ├── particle.py              # Single animated emoji particle (Particle)
│   ├── particle_system.py       # Spawns/updates/prunes particles (ParticleSystem)
│   ├── renderer.py              # PIL-based compositing: particles + glass panels (Renderer)
│   ├── ui.py                    # HUD stats overlay (UIOverlay)
│   └── utils.py                 # Math/timing helpers (FPSCounter, DeltaTimer, clamp, lerp...)
├── tests/
│   ├── test_camera.py
│   ├── test_expression.py
│   └── test_particle.py
├── app.py                       # Entry point: python app.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Emoji Font Setup

OpenCV cannot render color emoji glyphs natively, so Emoji Rain AI uses **Pillow** with a system or bundled emoji-capable font. On first run, the renderer searches (in order):

1. `assets/fonts/NotoColorEmoji.ttf` or `assets/fonts/Seguiemj.ttf` (drop a font here yourself)
2. Common OS install paths (Noto Color Emoji on Linux, Segoe UI Emoji on Windows, Apple Color Emoji on macOS)
3. A plain fallback font (emoji will render as empty boxes if no emoji font is found)

**For the best experience**, install a system emoji font:

- **Windows / macOS**: Segoe UI Emoji / Apple Color Emoji are pre-installed — no action needed.
- **Linux (Debian/Ubuntu)**: `sudo apt-get install fonts-noto-color-emoji`
- **Manual**: download `NotoColorEmoji.ttf` from the [Noto Emoji project](https://github.com/googlefonts/noto-emoji) and place it in `assets/fonts/`.

## Screenshots

> Add your own captures to `docs/screenshots/` and reference them here, e.g.:
>
> ![Smile detection with emoji rain](docs/screenshots/demo-smile.png)
>
> A `docs/demo.gif` walkthrough is a great addition for the top of this README once you have one recorded.

## Testing

```bash
pytest --cov=src --cov-report=term-missing
```

The suite covers:

- `tests/test_camera.py` — camera open/read/release lifecycle and reconnect logic, using a fake `VideoCapture` (no real hardware required)
- `tests/test_expression.py` — geometry primitives (EAR, MAR, smile intensity, eyebrow height) and the full expression decision tree, plus temporal-smoothing stability
- `tests/test_particle.py` — particle motion, rotation, fade-in/out, lifetime/expiry, and particle-system spawn/cap/prune behavior

## Future Improvements

- [ ] Multi-face support (currently tracks the primary detected face)
- [ ] Custom emoji packs / theming (swap the glyph set via config)
- [ ] Sound effects tied to expression changes (`assets/sounds/` is scaffolded and ready)
- [ ] On-screen calibration step to personalize thresholds to each user's neutral baseline
- [ ] Optional cloud-free desktop packaging (PyInstaller) for a one-click executable
- [ ] GPU-accelerated rendering path for very high particle counts

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -r requirements.txt`
3. Follow existing code style — `black` for formatting, `ruff` for linting, type hints and docstrings on all public functions/classes.
4. Add or update tests for any behavior change.
5. Run `black`, `ruff check`, `mypy`, and `pytest` locally before opening a PR — CI runs the same checks.
6. Open a pull request describing the change and why it's needed.

Please keep modules single-responsibility and avoid introducing magic numbers — add new thresholds to `src/config.py` instead.

## License

Distributed under the [MIT License](LICENSE).

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) by Google for the Face Mesh solution powering all landmark detection.
- [OpenCV](https://opencv.org/) for camera capture and core image operations.
- [Pillow](https://python-pillow.org/) for color emoji glyph rendering and glassmorphism compositing.
- The Eye Aspect Ratio technique is based on Soukupová & Čech, *"Real-Time Eye Blink Detection using Facial Landmarks"* (2016).
