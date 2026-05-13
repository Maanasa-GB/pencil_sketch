import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Pencil Sketch Generator",
    page_icon="✏️",
    layout="centered"
)

st.title("✏️ AI Pencil Sketch Generator")
st.write("Upload an image and convert it into a realistic pencil sketch.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- FUNCTIONS ----------------
def pencil_sketch(image, blur_value=21):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Invert image
    inverted = 255 - gray

    # Blur inverted image
    blurred = cv2.GaussianBlur(
        inverted,
        (blur_value, blur_value),
        0
    )

    # Invert blurred image
    inverted_blur = 255 - blurred

    # Create sketch
    sketch = cv2.divide(
        gray,
        inverted_blur,
        scale=256.0
    )

    # Improve contrast
    sketch = cv2.equalizeHist(sketch)

    return sketch


def color_sketch(image, sketch):

    # Convert sketch to 3 channels
    sketch_colored = cv2.cvtColor(
        sketch,
        cv2.COLOR_GRAY2RGB
    )

    # Blend with original image
    blend = cv2.addWeighted(
        image,
        0.25,
        sketch_colored,
        0.75,
        0
    )

    return blend

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    st.sidebar.header("Sketch Settings")

    blur = st.sidebar.slider(
        "Sketch Smoothness",
        5,
        51,
        21,
        step=2
    )

    # Generate sketch
    sketch = pencil_sketch(img, blur)

    # Generate color sketch
    color_version = color_sketch(img, sketch)

    # ---------------- OUTPUT ----------------
    st.subheader("Black & White Pencil Sketch")
    st.image(
        sketch,
        use_column_width=True,
        clamp=True
    )

    st.subheader("Colored Pencil Sketch")
    st.image(
        color_version,
        use_column_width=True
    )

    # ---------------- DOWNLOAD BUTTONS ----------------
    sketch_pil = Image.fromarray(sketch)

    color_pil = Image.fromarray(color_version)

    st.download_button(
        label="⬇ Download Pencil Sketch",
        data=sketch_pil.tobytes(),
        file_name="pencil_sketch.png",
        mime="image/png"
    )

    st.download_button(
        label="⬇ Download Colored Sketch",
        data=color_pil.tobytes(),
        file_name="colored_sketch.png",
        mime="image/png"
    )

else:
    st.info("Please upload an image to begin.")
