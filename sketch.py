import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Pencil Sketch Generator")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted_image = 255 - gray_image

    blurred = cv2.GaussianBlur(
        inverted_image,
        (21, 21),
        0
    )

    inverted_blurred = 255 - blurred

    sketch = cv2.divide(
        gray_image,
        inverted_blurred,
        scale=256.0
    )

    st.image(img, caption="Original Image")

    st.image(
        sketch,
        caption="Pencil Sketch",
        clamp=True
    )
