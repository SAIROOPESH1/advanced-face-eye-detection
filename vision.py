import cv2
import mediapipe as mp
from pathlib import Path
import urllib.request
import os

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
EYE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
)

if FACE_CASCADE.empty() or EYE_CASCADE.empty():
    raise RuntimeError(
        "OpenCV Haar cascade files failed to load. Ensure opencv-python(-headless) "
        "is installed correctly so cv2.data.haarcascades points to the bundled XML files."
    )

def ensure_model():
    """Download the official MediaPipe Face Landmarker model once if missing."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed. Download face_landmarker.task from Google's official "
                "MediaPipe model URL and place it in the models folder."
            ) from exc
    if MODEL_PATH.stat().st_size < 100_000:
        raise RuntimeError("The MediaPipe model file appears incomplete. Delete it and download again.")

def create_landmarker(mode="IMAGE"):
    ensure_model()
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    running_mode = (
        vision.RunningMode.VIDEO if mode == "VIDEO"
        else vision.RunningMode.IMAGE
    )
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=running_mode,
        num_faces=5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )
    return vision.FaceLandmarker.create_from_options(options)

def _blendshape_scores(result, face_index):
    scores = {}
    try:
        for item in result.face_blendshapes[face_index]:
            scores[item.category_name] = float(item.score)
    except (AttributeError, IndexError, TypeError):
        pass
    return scores

def analyze_frame(bgr, landmarker=None, mode="IMAGE", timestamp_ms=0,
                  use_haar=True, use_mesh=True, show_points=True, blink_threshold=0.55):
    output = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_count = 0
    eye_count = 0
    blink_count = 0

    if use_haar:
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.25, minNeighbors=5, minSize=(40, 40)
        )
        face_count = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 165, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray is not None and roi_gray.size > 0:
                eyes = EYE_CASCADE.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
                )
                eye_count += len(eyes)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(output, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    if use_mesh and landmarker is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if mode == "VIDEO":
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = landmarker.detect(mp_image)

        mesh_faces = result.face_landmarks or []
        if not use_haar:
            face_count = len(mesh_faces)

        h, w = output.shape[:2]
        for idx, landmarks in enumerate(mesh_faces):
            if show_points:
                # Draw a light sample of landmarks to keep the preview readable.
                for point_id in range(0, len(landmarks), 3):
                    point = landmarks[point_id]
                    px, py = int(point.x * w), int(point.y * h)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(output, (px, py), 1, (255, 190, 40), -1)

            scores = _blendshape_scores(result, idx)
            left_blink = scores.get("eyeBlinkLeft", 0.0)
            right_blink = scores.get("eyeBlinkRight", 0.0)
            is_blinking = left_blink >= blink_threshold or right_blink >= blink_threshold
            if is_blinking:
                blink_count += 1
            label = "BLINK" if is_blinking else "EYES OPEN*"
            anchor = landmarks[10]
            tx, ty = int(anchor.x * w), max(22, int(anchor.y * h) - 8)
            cv2.putText(output, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 220, 50), 2, cv2.LINE_AA)

    return output, {"faces": int(face_count), "eyes": int(eye_count), "blinks": int(blink_count)}
