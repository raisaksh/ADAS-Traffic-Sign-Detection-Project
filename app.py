import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_traffic_sign_model_fixed.keras')

def detect_and_crop_sign(image_path):
    """Detect red traffic sign using HSV and contours."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area = w * h
        aspect = w / h
        if area >= 500 and 0.5 < aspect < 1.5:
            cropped = img[y:y+h, x:x+w]
            cropped = cv2.resize(cropped, (64,64))
            return cropped, (x,y,w,h)
    # Fallback: use whole image as cropped sign
    cropped = cv2.resize(img, (64,64))
    return cropped, None

st.set_page_config(page_title="ADAS Traffic Sign Detection", layout="wide")
st.title("🚗 ADAS Traffic Sign Detection & Classification")
st.markdown("**Two-Stage Pipeline:** Detection → Classification")

st.write("**Step 1:** Upload a road scene image – the system detects and crops the traffic sign.")
st.write("**Step 2:** The cropped sign is classified into one of 43 German traffic sign categories.")

uploaded = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Detection
    cropped, bbox = detect_and_crop_sign(tmp_path)
    original = cv2.imread(tmp_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Scene")
        if bbox:
            x,y,w,h = bbox
            cv2.rectangle(original_rgb, (x,y), (x+w,y+h), (0,255,0), 2)
        st.image(original_rgb, width=300)  
    if cropped is not None:
        with col2:
            st.subheader("Detected Sign")
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), width=300)

        # Classification
        model = load_model()
        img_array = cropped / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_batch)[0]
        top3 = np.argsort(preds)[-3:][::-1]
        confidence = np.max(preds)

        with col3:
            st.subheader("Classification Result")
            st.metric("Predicted Class ID", str(top3[0]))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.write("**Top-3 Predictions:**")
            for i, cls in enumerate(top3):
                st.write(f"{i+1}. Class {cls} – {preds[cls]*100:.2f}%")
    else:
        st.error("No sign detected. Try another image.")