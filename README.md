# 😊 Emoji Rain AI

A real-time webcam app that detects your facial expression using MediaPipe Face Mesh and rains the matching emoji up the screen. No pretrained emotion classifier — expressions are computed from actual facial geometry (eye aspect ratio, mouth curvature, eyebrow height) using explicit, tunable rules.

```bash
python app.py
```

## Features

- 468-point facial landmark tracking via MediaPipe Face Mesh
- Geometry-based expression detection (no black-box ML model)
- Temporal smoothing so expressions don't flicker frame to frame
- Custom particle system for the floating emoji animation
- Glassmorphism HUD showing FPS, expression, confidence, particle count, etc.
- Live debug panel showing the raw numbers behind each decision
- Every threshold is configurable in one file
- 45 passing unit tests

## Installation

```bash
git clone https://github.com/acelin009/Emoji-Rain-AI.git
cd Emoji-Rain-AI

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Note:** MediaPipe's Windows wheels support Python 3.9–3.11. If install fails, check your Python version and use 3.11 if needed.

## Requirements

- Python 3.9–3.11
- A webcam
- On Linux, OpenCV needs `libgl1` and `libglib2.0-0`

## Usage

```bash
python app.py
```

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `h` | Toggle the HUD |
| `d` | Toggle the debug panel (on by default) |
| `l` | Toggle raw landmark overlay |

## Supported Expressions

| Expression | Emoji | Trigger |
|---|---|---|
| Neutral | 😐 | Default |
| Smile | 😊 | Mouth corners raised |
| Big Smile | 😁 | Strong smile + teeth visible |
| Laugh | 😂 | Mouth wide open + eyes narrowed |
| Wink | 😉 | One eye closed |
| Tongue Wink | 😜 | Wink + tongue out |
| Tongue Out | 😛 | Mouth open + tongue visible |
| Shocked | 😲 | Mouth open + eyebrows raised |
| Sad | 😢 | Mouth corners drooping |
| Angry | 😠 | Eyebrows furrowed |

## Configuration

Every tunable value lives in `src/config.py`:

- `CameraConfig` — resolution, FPS, mirror mode
- `LandmarkConfig` — MediaPipe detection settings
- `ExpressionThresholds` — all the geometry thresholds
- `EmojiMappingConfig` — expression → emoji/color mapping
- `ParticleConfig` — spawn rate, speed, scale, lifetime
- `UIConfig` — HUD colors, fonts, panel size

Edit a value, save, re-run.

## Calibration

Thresholds are tuned analytically, not against real camera data, so they may need adjusting for your face/lighting. Run the app (debug panel is on by default), hold each expression, and watch which value in the panel actually moves (`EAR`, `SMILE`, `EYEBROW`, `MOUTH OPEN`, etc.). Then adjust the matching threshold in `src/config.py`.

## Folder Structure

```
Emoji-Rain-AI/
├── src/
│   ├── camera.py                # Webcam capture
│   ├── landmark_detector.py     # MediaPipe wrapper
│   ├── geometry.py              # EAR, MAR, smile, eyebrow math
│   ├── expression_analyzer.py   # Classification + smoothing
│   ├── emoji_engine.py          # Expression → emoji mapping
│   ├── particle.py              # Single particle
│   ├── particle_system.py       # Spawns/updates particles
│   ├── renderer.py              # Drawing (particles + panels)
│   ├── ui.py                    # HUD + debug panel
│   ├── app_state.py             # Per-frame shared state
│   └── config.py                # All settings
├── tests/
├── app.py                       # Entry point
├── requirements.txt
└── README.md
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`** — run `pip install -r requirements.txt` inside your activated venv.

**Emoji show as empty boxes** — install a color emoji font. On Linux: `sudo apt-get install fonts-noto-color-emoji`. Windows/macOS already have one built in.

**An expression won't trigger** — open the debug panel (`d`) and check which number isn't crossing its threshold, then adjust it in `src/config.py`.

## Testing

```bash
pytest --cov=src --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

## 👨‍💻 Author

### **Acelin Nazareth**

AI & Data Science Engineer • Computer Vision • Machine Learning • Data Science

📧 **acelin.nazareth@gmail.com**

🐙 **GitHub:** https://github.com/acelin009

💼 **LinkedIn:** https://www.linkedin.com/in/acelin-nazareth-a7666a281/

⭐ If you enjoyed this project, consider giving it a star!

</div>
