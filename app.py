import streamlit as st
import numpy as np
import cv2
from model import NeuralNetwork
from streamlit_drawable_canvas import st_canvas

# Page Configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("MNIST Digit Recognizer")
st.markdown("""
Draw a digit (0-9) in the canvas below and click 'Predict' to see the model's prediction.
""")

# Sidebar for settings
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose Activation Function Model", 
                                    ("relu", "leaky_relu", "sigmoid", "tanh"))

# Load the corresponding model
@st.cache_resource
def load_cached_model(name):
    return NeuralNetwork.load_model(f"models/model_{name}.pkl")

try:
    model = load_cached_model(model_choice)
    st.sidebar.success(f"Loaded {model_choice} model successfully!")

except:
    st.sidebar.error(f"Failed to load {model_choice} model. Please ensure the model file exists.")

# Drawing Canvas
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="#000000",  # Black
        stroke_width=15,
        stroke_color="#FFFFFF",  # White
        background_color="#000000",  # Black
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Predict Logic
if canvas_result.image_data is not None:
    # 1. Get image data from canvas
    img = canvas_result.image_data.astype('uint8')
    # 2. Covert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # 4. Normalize
    img_normalized = img_resized / 255.0
    # 5. Flatten
    img_flattened = img_normalized.reshape(784, 1)

    with col2:
        st.write("### Model Input View:")
        st.image(img_resized, width=150)

    if st.button("Predict"):
        # Predict using the loaded model
        prediction = int(model.predict(img_flattened))

        # Get probability distribution (output layer A)
        probs = model.cache[f"A{model.L}"]
        confidence = probs[prediction] * 100

        st.metric(label="Predicted Digit", value=str(prediction))
        st.write(f"**Confidence:** {float(confidence):.2f}%")

        # Show probability chart
        st.bar_chart(probs.flatten())
        