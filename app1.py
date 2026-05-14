import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

disclaimer_markdown_text = """
**Disclaimer:** This model is for research and educational purposes only and should not be used for medical diagnosis. Always consult a qualified medical professional.
"""

num_classes = 4
class_mappings = {
    'Glioma': 0,
    'Meninigioma': 1,
    'Notumor': 2,
    'Pituitary': 3
}
inv_class_mappings = {v: k for k, v in class_mappings.items()}


# -----------------------------
# LOAD MODEL (cache so it loads once)
# -----------------------------

@st.cache_resource
def load_my_model():
    model_path = hf_hub_download(
        repo_id="hasibevnriaz/ML.brain_tumor_detect",  # 👈 CHANGE THIS
        filename="eff_model.keras"  # must match your HF file name
    )
    return load_model(model_path)

model = load_my_model()


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_tumor(img):
    if img is None:
        return None, None

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize
    img = cv2.resize(img, (224, 224))

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    # Prediction
    prediction = model.predict(img, verbose=0)[0]

    predictions_dict = {
        inv_class_mappings[i]: float(prediction[i])
        for i in range(num_classes)
    }

    return predictions_dict


# -----------------------------
# UI
# -----------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to classify it into tumor types.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        preds = predict_tumor(image)

        # -----------------------------
        # SHOW RESULTS
        # -----------------------------
        st.subheader("Prediction Results")

        st.json(preds)

        # Bar chart
        fig, ax = plt.subplots()
        classes = list(preds.keys())
        probs = list(preds.values())

        ax.bar(classes, probs)
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Probabilities")
        ax.set_ylabel("Probability")
        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)

        # Top prediction
        top_class = max(preds, key=preds.get)
        st.success(f"Predicted Class: {top_class}")

# Disclaimer
st.markdown(disclaimer_markdown_text)