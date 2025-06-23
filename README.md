# 🧠 Brain Tumor Detection Web App

This is a Streamlit app that detects brain tumors from MRI images using a pre-trained CNN model built in TensorFlow.

## How to Use

1. Upload an MRI image (`.jpg`, `.jpeg`, `.png`)
2. Click "Predict"
3. The app will tell you if a brain tumor is detected.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

The model (`brain_tumor_detector_model.h5`) was trained on a public brain tumor dataset using a 3-layer CNN.
