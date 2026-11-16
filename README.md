# Advanced Face & Eye Detection System

A Streamlit computer-vision application combining OpenCV Haar Cascade face/eye boxes with MediaPipe Face Landmarker facial landmarks and eye-blink blendshape scores.

## Features in this starter
- Live webcam processing with streamlit-webrtc
- Haar Cascade face and eye rectangles
- MediaPipe Face Landmarker points
- Eye blink indicator using `eyeBlinkLeft` and `eyeBlinkRight` blendshape scores
- Image upload and annotated-image download
- Session-only CSV report for image analyses
- FPS metric for the live stream
- No database and no image storage

## Recommended environment
Use **Python 3.11 (64-bit)** for a straightforward Windows setup with current MediaPipe wheels.

## Run on Windows in VS Code

Open this project folder in VS Code. In its terminal:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first run downloads the official MediaPipe Face Landmarker model into `models/face_landmarker.task`. An internet connection is needed for this first download. If automatic download fails, download the model from:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Save it as `models/face_landmarker.task`, then rerun the app.

## Notes
- Live webcam access is provided by WebRTC and may be blocked by strict firewalls or browser permissions.
- The blink label is based on MediaPipe blendshape scores and is an estimate. It is not a medical, driver-monitoring or safety system.
- This starter does not yet calculate head pose; implement and validate that as a later feature.
- The project detects facial features; it does not identify a person's identity.
- Webcam frames are processed in memory and are not written to disk by this app.

## Deployment
Push the project to GitHub and deploy the repository through Streamlit Community Cloud with `app.py` as the entry point. Confirm that the deployment environment supports MediaPipe and that WebRTC camera access works over HTTPS. Keep the model file out of Git if desired; the app downloads it on first startup.
