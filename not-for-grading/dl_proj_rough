# ================================================================
# COLORIZATION OF GRAYSCALE IMAGES USING DEEP NEURAL NETWORKS
# ================================================================

import os
import numpy as np
from PIL import Image
from skimage import color

import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
# Resize images to (256x256)
IMAGE_SIZE = 256   
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# DEFINE UTIL FUNCTIONS 
# ------------------------------------------------

def load_img(img_path):
    """
    Load an image from disk as a numpy RGB array.
    If grayscale, convert to 3-channel RGB by tiling.
    """
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:  
        # if grayscale → replicate channel 3 times
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=Image.BILINEAR):
    """
    Resize numpy image to (HW[0], HW[1]).
    """
    return np.asarray(
        Image.fromarray(img).resize((HW[1], HW[0]), resample=resample)
    )


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BILINEAR):
    """
    Preprocess image into L_orig (original size) and L_rs (resized).
    Returns:
      tens_orig_l : (1,1,H_orig,W_orig) -> original L channel tensor
      tens_rs_l   : (1,1,HW[0],HW[1]) -> resized L channel tensor
    """
    # Resize original
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert both original & resized images to LAB
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs   = color.rgb2lab(img_rgb_rs)

    # Extract only L channel
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs   = img_lab_rs[:, :, 0]

    # Convert to torch tensors with shape (1,1,H,W)
    tens_orig_l = torch.tensor(img_l_orig, dtype=torch.float32)[None, :, :]
    tens_rs_l   = torch.tensor(img_l_rs,   dtype=torch.float32)[None, :, :]

    #### Normalize L channel (paper recommends dividing by 100) -> I will look into this again(for now, lets have this) ####
    tens_orig_l = tens_orig_l / 100.0
    tens_rs_l   = tens_rs_l / 100.0

    return tens_orig_l, tens_rs_l


def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    """
    Combine predicted ab channels with original L channel and convert back to RGB.
    """
    HW_orig = tens_orig_l.shape[1:]          # (H_orig, W_orig)
    HW_pred = out_ab.shape[1:]               # (H_pred, W_pred)

    # If needed, resize ab to match original size
    if HW_pred != HW_orig:
        out_ab = F.interpolate(out_ab.unsqueeze(0), 
                               size=HW_orig, mode='bilinear')[0]

    # tens_orig_l: (1, H_orig, W_orig)
    # out_ab:      (2, H_orig, W_orig)

    out_lab = torch.cat((tens_orig_l, out_ab), dim=0)  # (3, H_orig, W_orig)

    out_lab_np = out_lab.cpu().numpy().transpose((1,2,0))
    out_rgb = color.lab2rgb(out_lab_np)

    return out_rgb

# ------------------------------------------------
# DEFINE DATASET CLASS 
# ------------------------------------------------

class ColorizationDataset(Dataset):
    """
    Dataset for automatic colorization.
    Loads images from folder, applies preprocessing,
    returns (L_resized, L_original, img_path).
    """
    def __init__(self, folder_path):
        self.image_paths = []

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(folder_path, fname))

        self.image_paths.sort()  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Load image to numpy RGB
        img_rgb_orig = load_img(img_path)

        # Apply preprocessing
        tens_orig_l, tens_rs_l = preprocess_img(
            img_rgb_orig,
            HW=(IMAGE_SIZE, IMAGE_SIZE)
        )

        # Return:
        # - resized L (input for model)
        # - original L (for reconstruction)
        # - image path
        return tens_rs_l, tens_orig_l, img_path

def colorization_collate(batch):
    """
    Allows batching resized L (fixed size),
    while keeping original L tensors in a list.
    """
    tens_rs_l_batch = torch.stack([b[0] for b in batch], dim=0)   # batchable
    tens_orig_l_list = [b[1] for b in batch]                     # NOT stacked
    img_paths = [b[2] for b in batch]                            # list of paths

    return tens_rs_l_batch, tens_orig_l_list, img_paths

# ------------------------------------------------
# DEFINE MODEL ARCHITECTURE & TRAINING LOOP
# ------------------------------------------------

def define_colorization_model():
    """
    TODO:
    - Build U-Net / encoder-decoder model
    - Output: predicted ab channels (1,2,256,256)
    """
    pass

def train_colorization_model():
    """
    TODO:
    Training loop:
    - Load ColorizationDataset
    - Forward pass
    - Compute loss (L1 / L2 / cross-entropy for bins)
    """
    pass


# ------------------------------------------------
# MAIN BLOCK 
# ------------------------------------------------

if __name__ == "__main__":
    # Folder containing your images
    DATA_FOLDER = "imagenet_50/train"  

    dataset = ColorizationDataset(DATA_FOLDER)

    loader = DataLoader(dataset, batch_size=4, shuffle=True,  collate_fn=colorization_collate)

#### Test the loader - keep this for now (just for me to check). ####
for tens_rs_l, tens_orig_l, img_paths in loader:
    print("\nBatch Loaded:")
    print("Resized L batch shape:", tens_rs_l.shape)  # (B, 1, 256, 256)

    print("Original L shapes:")
    for i, t in enumerate(tens_orig_l):
        print(f"  Image {i}: {t.shape}")

    print("Image paths:", img_paths)
    break

