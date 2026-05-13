import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cartoon Sketch Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Cartoon Sketch Generator")
st.write("Upload an image and convert it into a cartoon-style pencil sketch.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- FUNCTIONS ----------------
def cartoon_sketch(image):

    # Convert RGB to BGR
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Smooth colors while preserving edges
    color = cv2.bilateralFilter(img, 9, 250, 250)

    # Convert to grayscale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray_blur = cv2.medianBlur(gray, 5)

    # Create edge mask
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    # Cartoon effect
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Convert back to RGB
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

    return cartoon


def pencil_cartoon(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert image
    invert = 255 - gray

    # Blur inverted image
    blur = cv2.GaussianBlur(invert, (25, 25), 0)

    # Invert blur
    inverted_blur = 255 - blur

    # Pencil sketch
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    # Convert sketch to RGB
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    # Blend sketch with original image
    blend = cv2.addWeighted(
        image,
        0.35,
        sketch_rgb,
        0.65,
        0
    )

    return blend

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    st.sidebar.header("Filter Options")

    filter_type = st.sidebar.selectbox(
        "Choose Style",
        [
            "Cartoon Effect",
            "Cartoon Pencil Sketch"
        ]
    )

    if filter_type == "Cartoon Effect":
        output = cartoon_sketch(img)

    else:
        output = pencil_cartoon(img)

    st.subheader("✨ Generated Output")
    st.image(output, use_column_width=True)

    # ---------------- DOWNLOAD ----------------
    result = Image.fromarray(output)

    buf = io.BytesIO()
    result.save(buf, format="PNG")

    st.download_button(
        label="⬇ Download Image",
        data=buf.getvalue(),
        file_name="cartoon_sketch.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to generate a cartoon sketch.")
