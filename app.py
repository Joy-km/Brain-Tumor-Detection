import streamlit as st
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------
# 1. Custom CSS Theme & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Insert custom Google Font and advanced dark-mode styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global Body Overrides */
    .stApp {
        background-color: #0d0f18;
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }

    /* Header/Banner Section */
    .banner-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0d0f18 100%);
        border: 1px solid #3b0764;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(59, 7, 100, 0.25);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .banner-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .banner-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        background: linear-gradient(to right, #38bdf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .banner-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Sidebar Overrides */
    div[data-testid="stSidebar"] {
        background-color: #090a10;
        border-right: 1px solid #1e293b;
    }
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2 {
        color: #38bdf8;
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(22, 28, 45, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Target Box Highlights */
    .tag-glioma {
        background-color: rgba(244, 114, 182, 0.2);
        color: #f472b6;
        border: 1px solid #f472b6;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
    }
    .tag-meningioma {
        background-color: rgba(192, 132, 252, 0.2);
        color: #c084fc;
        border: 1px solid #c084fc;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
    }
    .tag-pituitary {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
    }
    .tag-notumor {
        background-color: rgba(74, 222, 128, 0.2);
        color: #4ade80;
        border: 1px solid #4ade80;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Cache Resource Loader & Helper Functions
# ----------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """Loads the best trained YOLOv8 model."""
    weights_path = "yolov8_brain_tumor_best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolov8s.pt"
    return YOLO(weights_path)

# Run inference
def predict_image(model, image, conf_threshold, iou_threshold, size=640):
    results = model.predict(source=image, imgsz=size, conf=conf_threshold, iou=iou_threshold, verbose=False)
    return results[0]

# ----------------------------------------------------
# 3. Model Loading
# ----------------------------------------------------
try:
    best_model = load_yolo_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading YOLO model: {e}")

# ----------------------------------------------------
# 4. Header Banner
# ----------------------------------------------------
st.markdown("""
<div class="banner-container">
    <div class="banner-title">🧠 NeuroScan AI</div>
    <div class="banner-subtitle">Deep Learning Dashboard for Brain Tumor Detection (YOLOv8)</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 5. Sidebar Layout
# ----------------------------------------------------
st.sidebar.title("🛠️ Configuration")

# Model Info Section
st.sidebar.markdown("""
<div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem; border-color: rgba(56, 189, 248, 0.4);">
    <h4 style="margin: 0; color: #38bdf8;">Model State</h4>
    <p style="font-size:0.85rem; margin:0.5rem 0 0 0; color: #94a3b8;">
        <b>Weights:</b> yolov8_brain_tumor_best.pt<br/>
        <b>Format:</b> PyTorch Module<br/>
        <b>Input Size:</b> 640x640px
    </p>
</div>
""", unsafe_allow_html=True)

# Detection Thresholds
st.sidebar.subheader("🔍 Detection Thresholds")
conf_threshold = st.sidebar.slider("Confidence Cutoff", min_value=0.10, max_value=1.00, value=0.25, step=0.05,
                                   help="Minimum score required to output a detection box.")
iou_threshold = st.sidebar.slider("NMS Overlap (IoU)", min_value=0.10, max_value=0.90, value=0.45, step=0.05,
                                  help="Threshold for removing overlapping prediction boxes.")

# ----------------------------------------------------
# 6. Main Panel Layout
# ----------------------------------------------------
# File uploader located directly in the main section
st.markdown("### 📷 Upload MRI Scan")
uploaded_file = st.file_uploader("Drag and drop or browse for an MRI slice image...", type=["png", "jpg", "jpeg"])

img_to_run = None
img_name = ""

if uploaded_file is not None:
    # Read custom uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_to_run = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_name = uploaded_file.name

# ----------------------------------------------------
# Prediction and Analysis View
# ----------------------------------------------------
if img_to_run is not None and model_loaded:
    # Run inference
    results = predict_image(best_model, img_to_run, conf_threshold, iou_threshold)

    # Parse boxes
    boxes = results.boxes
    num_detections = len(boxes)

    # Setup metrics
    max_conf = 0.0
    classes_detected = []
    if num_detections > 0:
        max_conf = float(boxes.conf.max().cpu().numpy())
        classes_detected = [best_model.names[int(cls.cpu().numpy())] for cls in boxes.cls]

    # Determine if any detection is an actual tumor 
    NO_TUMOR_LABELS = {"no tumor", "notumor", "no_tumor", "normal"}
    has_tumor = any(name.lower().replace(" ", "").replace("_", "") not in NO_TUMOR_LABELS
                    for name in classes_detected)

    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_detections}</div>
            <div class="metric-label">Objects Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_conf*100:.1f}%</div>
            <div class="metric-label">Max Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        status_text = "TUMOR DETECTED" if has_tumor else "NO TUMOR FOUND"
        status_color = "#f472b6" if has_tumor else "#4ade80"
        st.markdown(f"""
        <div class="metric-card" style="border-color: {status_color}50;">
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
            <div class="metric-label">Diagnostic Status</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        speed_ms = sum(results.speed.values())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{speed_ms:.1f} ms</div>
            <div class="metric-label">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Display images side-by-side
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("### 📷 Input MRI Scan")
        st.image(img_to_run, use_container_width=True, caption=f"Source: {img_name}")

    with col_img2:
        st.markdown("### 🎯 Detection Box Overlay")
        annotated_bgr = results.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True, caption="Model Annotations")

    # Bounding box details
    if num_detections > 0:
        st.markdown("### 📋 Detected Entities Table")

        rows_html = ""
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.cpu().numpy()[0])
            name = best_model.names[cls_id]
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            bbox_str = f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]"
            tag_key = name.lower().replace(" ", "")
            tag_color_map = {
                "glioma":      ("#f472b6", "rgba(244,114,182,0.15)"),
                "meningioma":  ("#c084fc", "rgba(192,132,252,0.15)"),
                "pituitary":   ("#38bdf8", "rgba(56,189,248,0.15)"),
                "notumor":     ("#4ade80", "rgba(74,222,128,0.15)"),
            }
            fg, bg = tag_color_map.get(tag_key, ("#e2e8f0", "rgba(255,255,255,0.1)"))
            tag_span = (f'<span style="background-color:{bg};color:{fg};border:1px solid {fg};'
                        f'border-radius:6px;padding:2px 10px;font-weight:600;">{name}</span>')
            rows_html += (
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<td style="padding:12px;">{i + 1}</td>'
                f'<td style="padding:12px;">{tag_span}</td>'
                f'<td style="padding:12px;font-weight:500;">{conf*100:.2f}%</td>'
                f'<td style="padding:12px;font-family:monospace;">{bbox_str}</td>'
                f'</tr>'
            )

        table_html = (
            '<table style="width:100%;border-collapse:collapse;text-align:left;'
            'background-color:rgba(22,28,45,0.3);border-radius:8px;overflow:hidden;">'
            '<thead><tr style="background-color:rgba(56,189,248,0.1);border-bottom:2px solid rgba(56,189,248,0.2);">'
            '<th style="padding:12px;font-weight:600;color:#38bdf8;">Index</th>'
            '<th style="padding:12px;font-weight:600;color:#38bdf8;">Tumor Class</th>'
            '<th style="padding:12px;font-weight:600;color:#38bdf8;">Confidence Score</th>'
            '<th style="padding:12px;font-weight:600;color:#38bdf8;">Bounding Box Location (px)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
        )
        # Use st.html() to render raw HTML without markdown interference
        st.html(table_html)
    else:
        st.success("🎉 No tumors detected in this MRI slice with the current confidence threshold.")

elif img_to_run is None:
    st.info("💡 Please upload an MRI scan image above to begin the detection pipeline.")

# ----------------------------------------------------
# Info & Safety Sections
# ----------------------------------------------------
st.write("")
st.markdown("---")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("### 🩺 Brain Tumor Dataset Details")
    st.write(
        "The model is trained on a stratified brain tumor MRI dataset consisting of 4 distinct classes representing various scanning outcomes:"
    )
    st.markdown("""
    - **Glioma**: A tumor that occurs in the brain and spinal cord, beginning in the glial cells that surround nerve cells.
    - **Meningioma**: A tumor that arises from the meninges — the membranes that surround your brain and spinal cord.
    - **Pituitary**: Abnormal growths that develop in your pituitary gland (often benign but highly visible).
    - **No Tumor**: Healthy MRI scan controls without tumorous growth or contrast abnormalities.
    """)

with col_info2:
    st.markdown("### 🚨 Safety & Clinical Disclaimer")
    st.warning("""
    **IMPORTANT CLINICAL DISCLAIMER:**

    This application and the underlying deep learning model are developed purely for **research and educational purposes**.
    - They are **NOT** medical diagnostic tools.
    - Do not use these results to replace clinical consultations, professional diagnoses, or medical decisions.
    - The bounding box coordinates represent statistical feature correlations and should not be used as actual surgical or clinical guides.
    """)
