import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import gdown

# ---------- Download model ----------
file_id = "1AvG7P-E2eoVQOQ4sMV4zWZfEEK9qDFY0"
url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
output = "mask_model.h5"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False, fuzzy=True)

model = load_model(output)

# ---------- Page configuration ----------
st.set_page_config(page_title="Mask Detection App", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Real-Time Mask Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use your camera to detect mask</p>", unsafe_allow_html=True)

# ---------- Helper function ----------
def detect_mask(model, img):
    img = cv2.resize(img, (224,224))
    img = img.astype('float32') / 255.0
    y_pred = model.predict(img.reshape(1,224,224,3))
    return "Mask" if y_pred[0][0] > 0.5 else "No Mask"

# ---------- Camera Input ----------
uploaded_image = st.camera_input("Take a picture")

if uploaded_image is not None:
    # Convert uploaded image to array
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Predict mask
    label = detect_mask(model, img)
    color = "#4CAF50" if label == "Mask" else "#F44336"
    
    # Display only the result
    st.markdown(f"<h2 style='text-align: center; color: {color};'>Prediction: {label}</h2>", unsafe_allow_html=True)
