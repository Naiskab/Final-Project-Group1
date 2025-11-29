# ================================================================
# COLORIZATION OF GRAYSCALE IMAGES USING DEEP NEURAL NETWORKS
# ================================================================

import os
import numpy as np
from PIL import Image
from skimage import color

import torch
import torch.nn as nn
import torch.nn.functional as F
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
      tens_rs_ab  : (2, HW[0], HW[1])    -> resized ab channels tensor
    """
    # Resize original
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert both original & resized images to LAB
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs   = color.rgb2lab(img_rgb_rs)

    # Extract only L channel
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs   = img_lab_rs[:, :, 0]

    # --- Extract ab channels from resized LAB ---
    img_ab_rs = img_lab_rs[:, :, 1:3]

    # Convert to torch tensors with shape (1,1,H,W)
    tens_orig_l = torch.tensor(img_l_orig, dtype=torch.float32)[None, :, :]
    tens_rs_l   = torch.tensor(img_l_rs,   dtype=torch.float32)[None, :, :]

    # ab: transpose to (2, H, W)
    tens_rs_ab = torch.tensor(img_ab_rs, dtype=torch.float32).permute(2, 0, 1)

    #### Normalize L channel (paper recommends dividing by 100) -> I will look into this again(for now, lets have this) ####
    tens_orig_l = tens_orig_l / 100.0
    tens_rs_l   = tens_rs_l / 100.0

    return tens_orig_l, tens_rs_l, tens_rs_ab


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
        tens_orig_l, tens_rs_l, tens_rs_ab = preprocess_img(
            img_rgb_orig,
            HW=(IMAGE_SIZE, IMAGE_SIZE)
        )

        # Return:
        # - resized L (input for model)
        # - original L (for reconstruction)
        # - image path
        return tens_rs_l, tens_rs_ab, tens_orig_l, img_path

def colorization_collate(batch):
    """
    Allows batching resized L (fixed size),
    while keeping original L tensors in a list.
    """
    tens_rs_l_batch = torch.stack([b[0] for b in batch], dim=0)   # batchable
    tens_rs_ab_batch = torch.stack([b[1] for b in batch], dim=0)
    tens_orig_l_list = [b[2] for b in batch]                     # NOT stacked
    img_paths = [b[3] for b in batch]                            # list of paths

    return tens_rs_l_batch, tens_rs_ab_batch, tens_orig_l_list, img_paths

# ------------------------------------------------
# DEFINE MODEL ARCHITECTURE & TRAINING LOOP
# ------------------------------------------------
class ColorizationUNet(nn.Module):
    """
    A simplified U-Net style encoder-decoder for colorization.
    Input:  (B, 1, 256, 256)   L channel
    Output: (B, 2, 256, 256)   predicted ab channels
    """
    def __init__(self):
        super().__init__()

        # ------------- Encoder -------------
        # Each block: conv - ReLU - conv - ReLU, then maxpool

        # Block 1: 1 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 256 -> 128

        # Block 2: 64 -> 128
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 128 -> 64

        # Block 3: 128 -> 256
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # 64 -> 32

        # Bottleneck: 256 -> 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ------------- Decoder -------------
        # Upconvs + skip connections from encoder

        # Up from 512 -> 256, concat with enc3 (256) -> 512
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Up from 256 -> 128, concat with enc2 (128) -> 256
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Up from 128 -> 64, concat with enc1 (64) -> 128
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final conv to 2 output channels (a and b)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, 64, 256, 256)
        p1 = self.pool1(e1)        # (B, 64, 128, 128)

        e2 = self.enc2(p1)         # (B, 128, 128, 128)
        p2 = self.pool2(e2)        # (B, 128, 64, 64)

        e3 = self.enc3(p2)         # (B, 256, 64, 64)
        p3 = self.pool3(e3)        # (B, 256, 32, 32)

        # Bottleneck
        b = self.bottleneck(p3)    # (B, 512, 32, 32)

        # Decoder with skip connections
        u3 = self.up3(b)           # (B, 256, 64, 64)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # (B, 256, 64, 64)

        u2 = self.up2(d3)          # (B, 128, 128, 128)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # (B, 128, 128, 128)

        u1 = self.up1(d2)          # (B, 64, 256, 256)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # (B, 64, 256, 256)

        out_ab = self.final_conv(d1)  # (B, 2, 256, 256)

        # For now, no activation; later we can normalize/scale ab if needed
        return out_ab

def define_colorization_model():
    """
    Build a simplified U-Net style colorization model.
    Input:  (B, 1, 256, 256)  L channel
    Output: (B, 2, 256, 256)  ab channels
    """
    model = ColorizationUNet().to(DEVICE)
    return model

def train_colorization_model(num_epochs=5):
    """
    Train the colorization U-Net on L - ab regression.
    Returns:
    model : The trained ColorizationUNet model.
    """
    # 1. Dataset & DataLoader
    dataset = ColorizationDataset("imagenet_50/train")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=colorization_collate,
    )

    # 2. Model, optimizer, loss
    model = define_colorization_model()  # already moves to DEVICE
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # L1 / MAE loss between predicted ab and ground-truth ab
    criterion = nn.L1Loss()

    # 3. Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_samples = 0

        for tens_rs_l, tens_rs_ab, _, _ in loader:
            # Move to device
            tens_rs_l = tens_rs_l.to(DEVICE)       # (B, 1, 256, 256)
            tens_rs_ab = tens_rs_ab.to(DEVICE)     # (B, 2, 256, 256)

            # Forward
            pred_ab = model(tens_rs_l)             # (B, 2, 256, 256)

            # Loss
            loss = criterion(pred_ab, tens_rs_ab)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            batch_size_curr = tens_rs_l.size(0)
            epoch_loss += loss.item() * batch_size_curr
            num_samples += batch_size_curr

        avg_loss = epoch_loss / max(1, num_samples)
        print(f"Epoch {epoch+1}/{num_epochs} - Train L1 loss: {avg_loss:.4f}")

    # 4. Save trained model weights
    torch.save(model.state_dict(), "colorization_unet.pth")
    print("Model weights saved to: colorization_unet.pth")

    return model

# ------------------------------------------------
# MAIN BLOCK
# ------------------------------------------------

if __name__ == "__main__":
    # Folder containing your images
    DATA_FOLDER = "imagenet_50/train"

    dataset = ColorizationDataset(DATA_FOLDER)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=colorization_collate
    )

    #### Test the loader - keep this for now (just for me to check). ####
    for tens_rs_l, tens_rs_ab, tens_orig_l, img_paths in loader:
        print("\nBatch Loaded:")
        print("Resized L batch shape:", tens_rs_l.shape)
        print("Resized ab batch shape:", tens_rs_ab.shape)

        print("Original L shapes:")
        for i, t in enumerate(tens_orig_l):
            print(f"  Image {i}: {t.shape}")

        print("Image paths:", img_paths)
        break

    # Only train when running this file directly
    model = train_colorization_model(num_epochs=5)
