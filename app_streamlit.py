import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import urllib.request

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# Disclaimer text
disclaimer_markdown_text = """
**Disclaimer:** This model is for research and educational purposes only and should not be used for medical diagnosis.
Always consult with a qualified medical professional for any health concerns.
The model's performance may vary based on image quality and specific pathological characteristics.
"""

# Class mappings
num_classes = 4
class_mappings = {
    'Glioma': 0,
    'Meninigioma': 1,
    'Notumor': 2,
    'Pituitary': 3
}
inv_class_mappings = {v: k for k, v in class_mappings.items()}



# Load Model (cached)
@st.cache_resource
def load_brain_tumor_model():
    MODEL_URL = "https://drive.google.com/uc?export=download&id=131_PF-QgBLKkpLngvOrz4mSVQkZvw5G8"
    MODEL_PATH = "eff_model.keras"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model file... Please wait."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_brain_tumor_model()


# -----------------------------
# Prediction Function
# -----------------------------
def predict_tumor(img):
    """
    img: numpy array (RGB or grayscale)
    returns:
        predictions_dict
        matplotlib figure
    """

    # If image is RGB, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    img = clahe.apply(img)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Add batch and channel dimensions
    # Final shape: (1, 224, 224, 1)
    img = np.expand_dims(img, axis=(0, -1))

    # Prediction
    prediction = model.predict(img, verbose=0)[0]

    # Convert to dictionary
    predictions_dict = {
        inv_class_mappings[i]: float(prediction[i])
        for i in range(num_classes)
    }

    # Create probability plot
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = list(predictions_dict.keys())
    probabilities = list(predictions_dict.values())

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    ax.bar(classes, probabilities, color=colors)
    ax.set_xlabel("Tumor Type")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return predictions_dict, fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI image to classify it into "
    "**Glioma, Meningioma, Pituitary tumor, or No Tumor**."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png","webp"]
)

if uploaded_file is not None:
    # Load image using PIL
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(
        pil_image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Convert to numpy array
    img = np.array(pil_image)

    # Prediction button
    if st.button("Predict"):
        with st.spinner("Analyzing MRI image..."):
            predictions_dict, fig = predict_tumor(img)

        # Sort predictions
        sorted_predictions = sorted(
            predictions_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_class, top_prob = sorted_predictions[0]

        # Display top prediction
        st.success(
            f"Prediction: **{top_class}** "
            f"({top_prob * 100:.2f}%)"
        )

        # Display all confidences
        st.subheader("Prediction with Confidence")
        for class_name, prob in sorted_predictions:
            st.write(f"**{class_name}:** {prob * 100:.2f}%")

        # Display probability chart
        st.subheader("Probability Distribution")
        st.pyplot(fig)

# Disclaimer
st.markdown("---")
st.markdown(disclaimer_markdown_text)