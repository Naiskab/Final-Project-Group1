import os
import sys
import numpy as np
from PIL import Image
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dl_proj_cnn import (
    load_img,
    preprocess_img,
    postprocess_tens,
    define_colorization_model,
    DEVICE,
    IMAGE_SIZE,
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

# model is inside parent folder
MODEL_PATH = os.path.abspath(os.path.join(PARENT_DIR, "colorization_unet.pth"))

# training images
TRAIN_DIR = os.path.abspath(os.path.join(PARENT_DIR, "imagenet_50/train"))


def load_trained_model(weights_path=MODEL_PATH):
    """
    Load the trained ColorizationUNet
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = define_colorization_model()
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def colorize_single_image(input_path, output_path="colorized_output.png"):
    """
    Colorize a single image using the trained U-Net model.
    """

    model = load_trained_model(MODEL_PATH)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img_rgb_orig = load_img(input_path)

    # preprocess (ignore ground-truth ab)
    tens_orig_l, tens_rs_l, _ = preprocess_img(
        img_rgb_orig,
        HW=(IMAGE_SIZE, IMAGE_SIZE),
    )

    tens_rs_l_batch = tens_rs_l.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_ab_batch = model(tens_rs_l_batch)

    pred_ab = pred_ab_batch[0].cpu()

    # L must return to [0,100] for LAB reconstruction
    tens_orig_l_lab = tens_orig_l * 100.0

    out_rgb = postprocess_tens(tens_orig_l_lab, pred_ab)

    out_rgb_uint8 = (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8)
    out_img = Image.fromarray(out_rgb_uint8)
    out_img.save(output_path)

    print(f"Saved colorized output to: {output_path}")


if __name__ == "__main__":

    # Pick the first train image as a demo
    example_input = None
    for fname in os.listdir(TRAIN_DIR):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            example_input = os.path.join(TRAIN_DIR, fname)
            break

    if example_input is None:
        raise RuntimeError("No images found in training directory!")

    print(f"Running inference on: {example_input}")
    colorize_single_image(example_input, "colorized_example.png")
