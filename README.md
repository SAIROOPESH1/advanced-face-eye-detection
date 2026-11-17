# 👁️ Advanced Face & Eye Detection System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://advanced-face-eye-detection.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-009688)](https://developers.google.com/mediapipe)

**Try the live app: [https://advanced-face-eye-detection.streamlit.app/](https://advanced-face-eye-detection.streamlit.app/)**

## Overview

A real-time computer-vision web application that detects faces and eyes in both live webcam streams and uploaded images. It fuses two complementary pipelines:

- **OpenCV Haar Cascades** — fast classical detection for face regions and eye regions inside each face.
- **MediaPipe Face Landmarker** — deep-learning-based 478-point 3D facial mesh with blendshape scores used for eye-blink estimation.

The two approaches run side by side with per-mode toggles, and every frame is annotated in memory — nothing is written to disk, making the app privacy-preserving by design.

## Key Features

- 🎥 **Live webcam processing** — real-time WebRTC video with face/eye boxes, landmark overlay, and a processing-FPS metric.
- 🖼️ **Static image analysis** — upload a JPG/PNG, view the annotated result, and download it.
- 🗺️ **Facial landmark mesh visualization** — up to 5 faces with 478 3D landmarks each, sampled for a readable preview.
- 🙂 **Eye-blink detection** — threshold-adjustable blink indicator built on MediaPipe `eyeBlinkLeft` / `eyeBlinkRight` blendshape scores.
- 🎛️ **Dual-mode detection toggles** — enable/disable Haar cascades, MediaPipe landmarks, and landmark rendering independently.
- 🔒 **Privacy-preserving** — all frame processing happens in memory; no frames or results are stored server-side.
- 📊 **Session report** — exportable CSV of image-analysis metrics for the current session.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web UI | Streamlit + streamlit-webrtc |
| Classical CV | OpenCV (Haar Cascade classifiers) |
| ML CV | MediaPipe Face Landmarker (mesh + blendshapes) |
| Data / math | NumPy, pandas |

## Project Structure

```
advanced-face-eye-detection/
├── app.py              # Streamlit UI: live camera, image analysis, guide
├── vision.py           # Detection pipeline: Haar cascades + MediaPipe
├── requirements.txt    # Python dependencies (opencv-python-headless)
├── packages.txt        # Streamlit Cloud system packages (libgl1, libgles2, libegl1, libglib2.0-0)
├── models/             # Auto-downloaded face_landmarker.task (git-ignored)
└── README.md
```

## Local Setup

Requires **Python 3.11 (64-bit)**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

On first run the app downloads the official MediaPipe Face Landmarker model (~4 MB) into `models/face_landmarker.task`. An internet connection is needed for this one-time download. If automatic download fails, fetch it manually from:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

and save it as `models/face_landmarker.task`.

## How It Works

1. **OpenCV Haar Cascade** locates face regions, then searches for eyes inside each detected face ROI.
2. **MediaPipe Face Landmarker** predicts 478 3D landmarks per face, with blendshape scores including `eyeBlinkLeft` and `eyeBlinkRight`.
3. Blink scores above the adjustable threshold are flagged as blinking.
4. The app draws face rectangles, eye rectangles, and a sampled landmark overlay, then reports counts and processing FPS.

## Deployment

Deployed on **Streamlit Community Cloud** — push to GitHub and point Streamlit Cloud at `app.py`. `packages.txt` provides the system libraries (`libgl1`, `libgles2`, `libegl1`, `libglib2.0-0`) that OpenCV and MediaPipe need in the cloud environment.

## Notes & Limitations

- The blink label is an estimate based on blendshape scores — not a medical, driver-monitoring, or safety system.
- Haar Cascade and MediaPipe may disagree; they use fundamentally different detection methods.
- WebRTC camera access requires HTTPS or localhost and may be blocked by strict firewalls.
- Performance depends on camera quality, lighting, CPU, and network.
- The project detects facial features; it does not identify a person's identity.

---

Built with Python · Streamlit · OpenCV · MediaPipe · NumPy
