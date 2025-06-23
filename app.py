import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI Scans")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_detector_model.h5')

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload image
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            st.error("❌ Brain Tumor Detected")
        else:
            st.success("✅ No Brain Tumor Detected")
