import streamlit as st
import cv2
import numpy as np
from face_recognition import faceRecognitionPipeline
import tempfile
from pathlib import Path

# ---------------------- Streamlit Config ----------------------
st.set_page_config(
    page_title="🎭 Beautiful AI — Face Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply a custom dark theme using markdown CSS
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0e1117;
            color: blue;
        }
        .stSidebar, .css-1d391kg {
            background-color: light blue !important;
        }
        .stMarkdown, .stText, .stSubheader {
            color: #f5f5f5 !important;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Title ----------------------
st.title("AI — Face Recognition")
st.markdown("Upload an **image**, **video**, or use your **webcam** to test the custom Face Recognition pipeline.")

# ---------------------- Sidebar ----------------------
st.sidebar.header("⚙️ Input Options")
mode = st.sidebar.radio("Select Input Type:", ["📷 Image", "🎞️ Video", "💻 Webcam"])

# ---------------------- Helper Function ----------------------
def display_predictions(pred_img, pred_dict):
    """Display predictions and intermediate outputs"""
    # Convert BGR → RGB safely
    rgb_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    st.image(rgb_img, caption="Predicted Image", width="stretch")

    st.subheader("🧠 Prediction Details")
    for i, d in enumerate(pred_dict):
        col1, col2 = st.columns(2)

        with col1:
            roi = np.array(d["roi"], dtype=np.float32)
            roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(roi, caption="Gray Image", channels="GRAY", width="stretch")

        with col2:
            eig_img = np.array(d["eig_image"], dtype=np.float32).reshape(100, 100)
            eig_img = cv2.normalize(eig_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(eig_img, caption="Eigen Image", channels="GRAY", width="stretch")

        st.markdown(f"**Prediction Gender:** {d['prediction_name']}")
        st.markdown(f"**Confidence:** {d['score'] * 100:.2f}%")
        st.divider()

# ---------------------- IMAGE MODE ----------------------
if mode == "📷 Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        tmp_path = Path(tempfile.gettempdir()) / uploaded_img.name
        tmp_path.write_bytes(uploaded_img.getbuffer())
        st.info(f"Running prediction on: {uploaded_img.name}")

        with st.spinner("Analyzing image..."):
            pred_img, pred_dict = faceRecognitionPipeline(str(tmp_path))

        display_predictions(pred_img, pred_dict)

# ---------------------- VIDEO MODE ----------------------
elif mode == "🎞️ Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        tmp_path = Path(tempfile.gettempdir()) / uploaded_vid.name
        tmp_path.write_bytes(uploaded_vid.getbuffer())
        st.info(f"Processing video: {uploaded_vid.name}")

        cap = cv2.VideoCapture(str(tmp_path))
        frame_placeholder = st.empty()

        with st.spinner("Running video predictions... (press Stop to end early)"):
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 5 == 0:  # Process every 5th frame for speed
                    pred_img, _ = faceRecognitionPipeline(frame, path=False)
                    rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, channels="RGB", caption=f"Frame {frame_count}", use_container_width=True)

                frame_count += 1

        cap.release()
        st.success("✅ Video processing complete!")

# ---------------------- WEBCAM MODE ----------------------
elif mode == "💻 Webcam":
    st.info("Capture a photo from your webcam below 👇")
    cam_photo = st.camera_input("Take a snapshot")
    if cam_photo:
        tmp_path = Path(tempfile.gettempdir()) / "webcam.jpg"
        tmp_path.write_bytes(cam_photo.getbuffer())

        with st.spinner("Analyzing webcam image..."):
            pred_img, pred_dict = faceRecognitionPipeline(str(tmp_path))

        display_predictions(pred_img, pred_dict)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("<center>🚀 Developed with using Streamlit + OpenCV</center>", unsafe_allow_html=True)
