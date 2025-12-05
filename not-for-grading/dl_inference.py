# evaluate_generator.py
# ================================================================
# EVALUATION OF COLORIZATION GENERATOR (NO DISCRIMINATOR)
# ================================================================

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo

from dl_rough_final_gan import (
    ECCVGenerator,
    FeatureExtractor,
    ColorizationDataset,
    postprocess_tens,
    IMAGE_SIZE,
    BATCH_SIZE
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_DISTANCE = 150.0      # threshold sweep 0..150 (as they did in the paper)
N_BINS = 151

def evaluate_generator(
    gen_weights_path="final_gan_generator.pth",
    data_folder="../imagenet_50/validation",
    batch_size=None,
    save_examples=True,
    examples_dir="gen_eval_examples",
    max_batches=None,
):
    """
    Evaluates the saved generator ONLY.
    Computes:
        - L1 loss
        - MSE loss
        - Perceptual (content) loss (using VGG)
        - Raw AuC (as in Zhang et al. 2016): area under cumulative
          distribution of per-pixel L2 color error in ab space, 0..150
    """

    if batch_size is None:
        batch_size = BATCH_SIZE

    # -------------------------------------------------------
    # 1. Load generator
    # -------------------------------------------------------
    # ---------------------------------------
    # Finetuned generator
    # ---------------------------------------
    # print(f"Loading generator from: {gen_weights_path}")
    # generator = ECCVGenerator().to(DEVICE)
    # generator.load_state_dict(torch.load(gen_weights_path, map_location=DEVICE))
    # generator.eval()

    #----------------------------------------
    # pretrained model
    #---------------------------------------
    print("Loading pretrained ECCV model from Zhang et al. (2016)...")

    generator = ECCVGenerator()
    generator.load_state_dict(
        model_zoo.load_url(
            'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
            map_location='cpu',
            check_hash=True
        )
    )
    generator.to(DEVICE)
    generator.eval()
    # ---------------------------------------
    #
    # ------------------------------------------------------

    # Feature extractor (for perceptual loss)
    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    # Losses
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()
    perceptual_loss_fn = nn.L1Loss()

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------
    eval_dataset = ColorizationDataset(data_folder, mode='training')
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset size: {len(eval_dataset)}")

    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)

    # -------------------------------------------------------
    # Accumulate losses + error histogram for AuC
    # -------------------------------------------------------
    total_l1 = 0.0
    total_mse = 0.0
    total_perc = 0.0
    n_batches = 0

    # Histogram over error distances [0, MAX_DISTANCE]
    error_hist = torch.zeros(N_BINS, dtype=torch.float64)
    total_pixels = 0

    with torch.no_grad():
        for batch_idx, (tens_l, tens_ab_gt) in enumerate(tqdm(eval_loader)):
            tens_l = tens_l.to(DEVICE)
            tens_ab_gt = tens_ab_gt.to(DEVICE)

            # Run generator
            pred_ab = generator(tens_l)

            # -----------------------
            # Compute losses
            # -----------------------
            l1_loss_val = l1_loss_fn(pred_ab, tens_ab_gt).item()
            mse_loss_val = mse_loss_fn(pred_ab, tens_ab_gt).item()

            fake_lab = torch.cat([tens_l, pred_ab], dim=1)
            real_lab = torch.cat([tens_l, tens_ab_gt], dim=1)
            perc_loss_val = perceptual_loss_fn(
                feature_extractor(fake_lab),
                feature_extractor(real_lab)
            ).item()

            total_l1 += l1_loss_val
            total_mse += mse_loss_val
            total_perc += perc_loss_val
            n_batches += 1

            # -----------------------
            # Update AuC histogram
            # -----------------------
            # pred_ab, tens_ab_gt: (B, 2, H, W)
            diff = pred_ab - tens_ab_gt
            # L2 error per pixel in ab space (B, H, W)
            err = torch.sqrt(torch.sum(diff ** 2, dim=1))

            # Flatten and move to CPU
            err_flat = err.reshape(-1).cpu()

            # Clamp to [0, MAX_DISTANCE] so all errors > MAX_DISTANCE
            # go into the last bin
            err_flat = torch.clamp(err_flat, 0.0, MAX_DISTANCE)

            # Histogram with N_BINS bins from 0 to MAX_DISTANCE
            # bin i corresponds to threshold i (0..150 if MAX_DISTANCE=150)
            batch_hist = torch.histc(
                err_flat,
                bins=N_BINS,
                min=0.0,
                max=MAX_DISTANCE
            )

            error_hist += batch_hist
            total_pixels += err_flat.numel()

            # -----------------------
            # Save a few examples
            # -----------------------
            if save_examples and batch_idx < 5:
                for i in range(min(2, tens_l.size(0))):
                    gray = tens_l[i].cpu()
                    real_ab = tens_ab_gt[i].cpu()
                    fake_ab = pred_ab[i].cpu()

                    gray_rgb = postprocess_tens(gray, torch.zeros_like(real_ab))
                    real_rgb = postprocess_tens(gray, real_ab)
                    fake_rgb = postprocess_tens(gray, fake_ab)

                    comparison = np.concatenate([gray_rgb, real_rgb, fake_rgb], axis=1)
                    Image.fromarray((comparison * 255).astype(np.uint8)).save(
                        os.path.join(examples_dir, f"batch{batch_idx}_img{i}.png")
                    )

            if max_batches is not None and n_batches >= max_batches:
                break

    # -------------------------------------------------------
    # Final scalar losses
    # -------------------------------------------------------
    avg_l1 = total_l1 / n_batches
    avg_mse = total_mse / n_batches
    avg_perc = total_perc / n_batches

    # -------------------------------------------------------
    # Compute AuC from histogram
    # -------------------------------------------------------
    if total_pixels > 0:
        # Normalize histogram: probability per bin
        pdf = error_hist / float(total_pixels)  # shape (N_BINS,)
        # Cumulative mass function over thresholds 0..MAX_DISTANCE
        cmf = torch.cumsum(pdf, dim=0)         # shape (N_BINS,)

        # x-axis values: thresholds 0..MAX_DISTANCE
        thresholds = torch.linspace(0.0, MAX_DISTANCE, steps=N_BINS)
        # Area under CMF curve using trapezoidal rule
        auc_raw = torch.trapz(cmf, thresholds).item()
        # Normalize so that max possible AuC is 1 (integral of 1 dt from 0..MAX_DISTANCE)
        auc_norm = auc_raw / MAX_DISTANCE
    else:
        auc_norm = float('nan')

    print("\n===== GENERATOR EVALUATION RESULTS =====")
    print(f"Avg L1 Loss:         {avg_l1:.4f}")
    print(f"Avg MSE Loss:        {avg_mse:.4f}")
    print(f"Avg Perceptual Loss: {avg_perc:.4f}")
    print(f"Raw AuC (0..{int(MAX_DISTANCE)}): {auc_norm:.4f}")
    print("========================================\n")

    return {
        "avg_l1": avg_l1,
        "avg_mse": avg_mse,
        "avg_perceptual": avg_perc,
        "auc_raw_norm": auc_norm
    }



if __name__ == "__main__":
    results = evaluate_generator(
        gen_weights_path="best_gan_generator.pth",
        data_folder="../imagenet_50/validation",
        batch_size=16,
        save_examples=True,
        examples_dir="gen_eval_examples",
        max_batches=None,
    )
