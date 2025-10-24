import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# ===================================================
# PAGE CONFIGURATION
# ===================================================
st.set_page_config(
    page_title="Garbage Detection AI",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===================================================
# CUSTOM STYLING
# ===================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00A884;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .metric-card {
        background-color: #111827;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 6px solid #00A884;
        color: #E5E7EB;
    }
    .detection-box {
        background: linear-gradient(135deg, #00A884, #007A63);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.6rem 0;
    }
    .info-box {
        background-color: #E8F9F1;
        color: #1B4332;
        padding: 1rem;
        border-radius: 10px;
        border-left: 6px solid #00A884;
        margin: 1rem 0;
    }
    .performance-metric {
        background: linear-gradient(135deg, #00A884, #007A63);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================================================
# MODEL LOADING
# ===================================================
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.title("⚙️ Configuration")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.4, 0.05,
    help="Higher = fewer false alarms, Lower = find more garbage"
)

model = load_model()
if model:
    st.sidebar.success("✅ Model loaded successfully!")

# ===================================================
# HEADER
# ===================================================
st.markdown('<h1 class="main-header">🗑️ Garbage Detection on Streets</h1>', unsafe_allow_html=True)
st.markdown("### Detect garbage in images and videos using YOLOv8")

# ===================================================
# MAIN TABS
# ===================================================
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 Model Info"])

# ===================================================
# IMAGE DETECTION TAB - SIDE-BY-SIDE LAYOUT
# ===================================================
with tab1:
    st.header("Image Detection")

    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image and model:
        image = Image.open(uploaded_image)
        image_cv = np.array(image)
        if image_cv.shape[-1] == 4:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)
        else:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Centered Detect Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            detect_button = st.button("🔍 Detect Garbage", type="primary", use_container_width=True)

        if detect_button:
            with st.spinner("🔍 Analyzing image for garbage..."):
                results = model(image_cv, conf=confidence)
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # SIDE-BY-SIDE DISPLAY
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="📸 Original Image", use_container_width=True)
                with col2:
                    st.image(annotated_rgb, caption="🗑️ Detected Garbage", use_container_width=True)

                # DETECTION RESULTS BELOW
                st.markdown("---")
                boxes = results[0].boxes

                if boxes and len(boxes) > 0:
                    st.success(f"🎯 Found {len(boxes)} garbage objects!")

                    cols = st.columns([1, 2, 1])
                    with cols[1]:
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            conf_score = float(box.conf[0])
                            st.markdown(f"""
                            <div class="detection-box">
                                <b>🗑️ Detection {i+1}</b><br>
                                📈 Confidence: {conf_score:.3f}<br>
                                📊 Class: {model.names[cls_id]}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("❌ No garbage detected in this image")

# ===================================================
# VIDEO DETECTION TAB
# ===================================================
with tab2:
    st.header("Video Detection")

    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])

    if uploaded_video and model:
        st.markdown("<div class='center-content'>", unsafe_allow_html=True)
        st.video(uploaded_video)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_video = st.button("🚀 Process Video", type="primary", use_container_width=True)

        if process_video:
            with st.spinner("Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_video.read())
                    temp_path = tfile.name

                cap = cv2.VideoCapture(temp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)
                status = st.empty()

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=confidence, verbose=False)
                    annotated = results[0].plot()
                    out.write(annotated)

                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                    status.text(f"Processed {frame_count}/{total_frames} frames")

                cap.release()
                out.release()
                os.unlink(temp_path)

                st.markdown("<div class='center-content'>", unsafe_allow_html=True)
                st.success("✅ Processing completed!")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        "📥 Download Processed Video",
                        f,
                        "garbage_detected.mp4",
                        "video/mp4"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# MODEL INFO TAB
# ===================================================
with tab3:
    st.header("Model Information")

    if model:
        st.markdown("<div class='center-content'>", unsafe_allow_html=True)

        st.subheader("Model Details")
        st.markdown(f"""
        <div class="metric-card">
            🤖 <b>Model:</b> YOLOv8n<br>
            🎯 <b>Task:</b> Object Detection<br>
            🗑️ <b>Class:</b> Garbage<br>
            📐 <b>Input Size:</b> 640x640<br>
            ⚙️ <b>Confidence:</b> {confidence}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Performance Metrics")
        cols = st.columns(4)
        metrics = [
            ("89.4%", "Precision"),
            ("80.8%", "Recall"),
            ("86.4%", "mAP50"),
            ("150ms", "Speed")
        ]
        for col, (value, label) in zip(cols, metrics):
            with col:
                st.markdown(f'''
                <div class="performance-metric">
                    <h4>{value}</h4>
                    <p>{label}</p>
                </div>
                ''', unsafe_allow_html=True)

        st.subheader("Usage Tips")
        st.markdown('''
        <div class="info-box">
            <b>🔄 Confidence Guide:</b><br><br>
            • <b>0.5-0.6:</b> Clean areas (fewer false alarms)<br>
            • <b>0.4:</b> Balanced (default setting)<br>
            • <b>0.3-0.4:</b> Littered areas (find more garbage)<br>
            • <b>0.1-0.2:</b> Maximum detection
        </div>
        ''', unsafe_allow_html=True)

        st.subheader("Technical Details")
        st.markdown('''
        <div class="metric-card">
            <b>📊 Training Information:</b><br><br>
            • <b>Dataset:</b> Custom Garbage Collection<br>
            • <b>Training Epochs:</b> 100<br>
            • <b>Training Images:</b> ~3,200<br>
            • <b>Framework:</b> Ultralytics YOLOv8<br>
            • <b>Backbone:</b> CSPDarknet
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("❌ Model failed to load")

# ===================================================
# FOOTER
# ===================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #00A884;'>"
    "<b>Built with ❤️ using Streamlit & YOLOv8 by Shailesh V</b>"
    "</div>",
    unsafe_allow_html=True
)
