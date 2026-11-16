import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from vision import create_landmarker, analyze_frame, MODEL_PATH, ensure_model

st.set_page_config(page_title="Advanced Face & Eye AI", page_icon="👁️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(120,150,200,.25);
       border-radius: 16px; background: linear-gradient(110deg, rgba(25,50,85,.75), rgba(20,28,45,.65));}
.small-muted {color: #9aa9bd; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>👁️ Advanced Face & Eye Detection</h1><p class="small-muted">Classical computer vision + MediaPipe facial landmarks and blendshapes</p></div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("⚙️ Detection settings")
    use_haar = st.checkbox("Enable Haar face & eye boxes", value=True)
    use_mesh = st.checkbox("Enable MediaPipe landmarks", value=True)
    show_points = st.checkbox("Show facial landmark points", value=True)
    blink_threshold = st.slider("Blink threshold", 0.10, 0.90, 0.55, 0.05)
    st.caption("A higher threshold requires a stronger eye-blink score.")
    st.divider()
    st.caption("Privacy: frames are processed in memory. This app does not save webcam frames.")

@st.cache_resource
def get_image_landmarker():
    ensure_model()
    return create_landmarker("IMAGE")

try:
    if use_mesh:
        ensure_model()
except Exception as exc:
    st.error(f"Could not prepare the MediaPipe model: {exc}")
    st.markdown(f"Download the model manually from [Google's official model file]({MODEL_PATH.as_uri() if MODEL_PATH.exists() else 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'}) and save it as `models/face_landmarker.task`.")
    st.stop()

tab_live, tab_image, tab_guide = st.tabs(["📷 Live camera", "🖼️ Image analysis", "📘 How it works"])

if "history" not in st.session_state:
    st.session_state.history = []

with tab_live:
    st.subheader("Live webcam")
    st.info("Press START. Allow camera access when your browser asks. Live camera streaming requires localhost or HTTPS.")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        class LiveProcessor(VideoProcessorBase):
            def __init__(self):
                self.landmarker = create_landmarker("VIDEO") if use_mesh else None
                self.last_timestamp = 0
                self.face_count = 0
                self.eye_count = 0
                self.blink_count = 0
                self.fps = 0.0
                self.last_time = time.perf_counter()
                self.frame_counter = 0

            def recv(self, frame):
                bgr = frame.to_ndarray(format="bgr24")
                now = time.perf_counter()
                self.frame_counter += 1
                elapsed = now - self.last_time
                if elapsed >= 1.0:
                    self.fps = self.frame_counter / elapsed
                    self.frame_counter = 0
                    self.last_time = now

                annotated, metrics = analyze_frame(
                    bgr, self.landmarker, "VIDEO", self.last_timestamp + 1,
                    use_haar, use_mesh, show_points, blink_threshold
                )
                self.last_timestamp += 1
                self.face_count = metrics["faces"]
                self.eye_count = metrics["eyes"]
                self.blink_count = metrics["blinks"]
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        ctx = webrtc_streamer(
            key="advanced-face-eye",
            video_processor_factory=LiveProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            p = ctx.video_processor
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faces", p.face_count)
            m2.metric("Eyes (Haar)", p.eye_count)
            m3.metric("Blinking faces", p.blink_count)
            m4.metric("Processing FPS", f"{p.fps:.1f}")
    except Exception as exc:
        st.error(f"Live camera component could not start: {exc}")

with tab_image:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded:
        raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("The uploaded file could not be decoded as an image.")
        else:
            if st.button("Analyze image", type="primary"):
                landmarker = get_image_landmarker() if use_mesh else None
                annotated, metrics = analyze_frame(
                    bgr, landmarker, "IMAGE", 0,
                    use_haar, use_mesh, show_points, blink_threshold
                )
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Analysis result", use_container_width=True)
                a, b, c = st.columns(3)
                a.metric("Faces detected", metrics["faces"])
                b.metric("Eyes detected (Haar)", metrics["eyes"])
                c.metric("Blinking faces", metrics["blinks"])
                st.session_state.history.append({
                    "Time": time.strftime("%H:%M:%S"),
                    "Faces": metrics["faces"],
                    "Eyes": metrics["eyes"],
                    "Blinking faces": metrics["blinks"]
                })
                ok, encoded = cv2.imencode(".jpg", annotated)
                if ok:
                    st.download_button("⬇️ Download annotated image", encoded.tobytes(),
                                       "face_eye_analysis.jpg", "image/jpeg")

with tab_guide:
    st.subheader("Detection pipeline")
    st.markdown("""
    1. **OpenCV Haar Cascade** searches for face regions and then eyes inside each face.
    2. **MediaPipe Face Landmarker** predicts 478 3D facial landmarks per detected face.
    3. MediaPipe blendshape scores include `eyeBlinkLeft` and `eyeBlinkRight`; these scores are used as a blink indicator.
    4. The app draws face rectangles, eye rectangles and selected landmark points.
    5. The dashboard reports counts and approximate processing FPS.

    **Important limitations**
    - A blink score is an estimate, not a medical or safety-grade measurement.
    - Head pose is not yet calculated in this first advanced build; we will add and validate it as a separate stage.
    - Haar Cascade and MediaPipe may disagree because they use different detection methods.
    - Performance depends on camera quality, lighting, CPU and network.
    """)
    if st.session_state.history:
        st.markdown("### Image analysis history (current session)")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download session report (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "detection_report.csv", "text/csv")
    else:
        st.caption("Analyze an image to start a session report.")

st.divider()
st.caption("Built with Python · Streamlit · OpenCV · MediaPipe · NumPy")
