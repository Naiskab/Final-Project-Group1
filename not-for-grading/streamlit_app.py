import streamlit as st
import torch
import numpy as np
from PIL import Image
from dl_proj_pretrained_gan import ECCVGenerator, preprocess_img, postprocess_tens

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = ECCVGenerator().to(DEVICE)
    state = torch.load("best_gan_generator.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

generator = load_model()

st.title("Image Colorizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load original image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    tens_orig_l, tens_rs_l = preprocess_img(img_np, HW=(256, 256), return_ab=False)
    tens_rs_l = tens_rs_l.unsqueeze(0).to(DEVICE)  # (1,1,256,256)

    # inference
    with torch.no_grad():
        pred_ab = generator(tens_rs_l)[0].cpu()  # (2,H,W)

    # postprocess
    colorized = postprocess_tens(tens_orig_l, pred_ab)
    colorized_img = (colorized * 255).astype(np.uint8)

    # display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img_np)

    with col2:
        st.subheader("Colorized Output")
        st.image(colorized_img)
