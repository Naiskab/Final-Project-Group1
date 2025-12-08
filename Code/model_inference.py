import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Import components from the finetuned GAN training script
from dl_rough_final_gan import (
    ECCVGenerator,
    FeatureExtractor,
    ColorizationDataset,
    postprocess_tens,
    BATCH_SIZE
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_DISTANCE = 150.0
N_BINS = 151


def evaluate_single_model(
    generator: nn.Module,
    eval_loader: DataLoader,
    feature_extractor: nn.Module,
):
    """
    Run evaluation for a *single* generator instance.
    Returns a dict of scalar metrics.
    """

    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()
    perceptual_loss_fn = nn.L1Loss()

    total_l1 = 0.0
    total_mse = 0.0
    total_perc = 0.0
    n_batches = 0

    total_psnr = 0.0
    total_psnr_count = 0

    total_ssim = 0.0
    total_ssim_count = 0

    error_hist = torch.zeros(N_BINS, dtype=torch.float64)
    total_pixels = 0

    generator.eval()
    feature_extractor.eval()

    with torch.no_grad():
        for batch_idx, (tens_l, tens_ab_gt) in enumerate(tqdm(eval_loader)):
            tens_l = tens_l.to(DEVICE)
            tens_ab_gt = tens_ab_gt.to(DEVICE)

            # Forward pass
            pred_ab = generator(tens_l)

            # ---------------------------------------------------
            # Losses in ab space
            # ---------------------------------------------------
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

            # ---------------------------------------------------
            # PSNR & SSIM in RGB space (values in [0, 1])
            # ---------------------------------------------------
            for i in range(tens_l.size(0)):
                gray_cpu = tens_l[i].cpu()
                real_ab_cpu = tens_ab_gt[i].cpu()
                fake_ab_cpu = pred_ab[i].cpu()

                real_rgb = postprocess_tens(gray_cpu, real_ab_cpu)
                fake_rgb = postprocess_tens(gray_cpu, fake_ab_cpu)

                # PSNR with MAX=1.0
                mse_rgb = np.mean((fake_rgb - real_rgb) ** 2)
                if mse_rgb > 0:
                    psnr_val = 10.0 * np.log10(1.0 / mse_rgb)
                else:
                    psnr_val = float("inf")

                total_psnr += psnr_val
                total_psnr_count += 1

                # SSIM (skimage expects HWC)
                ssim_val = ssim(
                    real_rgb,
                    fake_rgb,
                    data_range=1.0,
                    channel_axis=-1,
                )
                total_ssim += ssim_val
                total_ssim_count += 1

            n_batches += 1

            # ---------------------------------------------------
            # AUC histogram in ab space
            # ---------------------------------------------------
            diff = pred_ab - tens_ab_gt             # (B,2,H,W)
            err = torch.sqrt(torch.sum(diff ** 2, dim=1))  # (B,H,W)
            err_flat = err.reshape(-1).cpu()
            err_flat = torch.clamp(err_flat, 0.0, MAX_DISTANCE)

            batch_hist = torch.histc(
                err_flat,
                bins=N_BINS,
                min=0.0,
                max=MAX_DISTANCE,
            )

            error_hist += batch_hist
            total_pixels += err_flat.numel()

    # -----------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------
    avg_l1 = total_l1 / n_batches
    avg_mse = total_mse / n_batches
    avg_perc = total_perc / n_batches

    avg_psnr = total_psnr / total_psnr_count if total_psnr_count > 0 else float("nan")
    avg_ssim = total_ssim / total_ssim_count if total_ssim_count > 0 else float("nan")

    if total_pixels > 0:
        pdf = error_hist / float(total_pixels)
        cmf = torch.cumsum(pdf, dim=0)

        thresholds = torch.linspace(0.0, MAX_DISTANCE, steps=N_BINS)
        auc_raw = torch.trapz(cmf, thresholds).item()
        auc_norm = auc_raw / MAX_DISTANCE
    else:
        auc_norm = float("nan")

    return {
        "avg_l1": avg_l1,
        "avg_mse": avg_mse,
        "avg_perceptual": avg_perc,
        "auc_raw_norm": auc_norm,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "num_batches": n_batches,
    }


def evaluate_both_models(
    gan_weights_path="gan_generator_epoch_50.pth",
    split="validation",         
    batch_size=None,
    max_batches=None,
):
    """
    Evaluate BOTH:
      1) Fine-tuned GAN generator
      2) Pretrained ECCV-16 generator

    on the same Hugging Face split (e.g., 'train' or 'validation').
    """

    if batch_size is None:
        batch_size = BATCH_SIZE

    # -------------------------------------------------------
    # Dataset / dataloader (Hugging Face via ColorizationDataset)
    eval_dataset = ColorizationDataset(split=split, mode="training")

    if max_batches is not None:
        max_samples = max_batches * batch_size
        eval_dataset = torch.utils.data.Subset(
            eval_dataset,
            range(min(len(eval_dataset), max_samples))
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset size for evaluation (split='{split}'): {len(eval_dataset)}")

    # -------------------------------------------------------
    # Shared feature extractor
    # -------------------------------------------------------
    feature_extractor = FeatureExtractor().to(DEVICE)

    results = {}

    # -------------------------------------------------------
    # Fine-tuned GAN generator
    # -------------------------------------------------------
    print(f"\n=== Evaluating Fine-tuned GAN generator ({gan_weights_path}) ===")
    gan_gen = ECCVGenerator().to(DEVICE)
    gan_gen.load_state_dict(torch.load(gan_weights_path, map_location=DEVICE))

    gan_results = evaluate_single_model(gan_gen, eval_loader, feature_extractor)
    results["finetuned_gan"] = gan_results

    print("===== FINETUNED GAN RESULTS =====")
    print(f"Avg L1 Loss:         {gan_results['avg_l1']:.4f}")
    print(f"Avg MSE Loss:        {gan_results['avg_mse']:.4f}")
    print(f"Avg Perceptual Loss: {gan_results['avg_perceptual']:.4f}")
    print(f"Raw AUC (0..{int(MAX_DISTANCE)}): {gan_results['auc_raw_norm']:.4f}")
    print(f"Avg PSNR:            {gan_results['avg_psnr']:.4f}")
    print(f"Avg SSIM:            {gan_results['avg_ssim']:.4f}")
    print(f"Total batches:       {gan_results['num_batches']}")
    print("=================================\n")

    # -------------------------------------------------------
    # Pretrained ECCV-16 generator (Zhang et al.)
    # -------------------------------------------------------
    print("=== Evaluating Pretrained ECCV-16 generator (Zhang et al., 2016) ===")
    pre_gen = ECCVGenerator().to(DEVICE)
    pre_gen.load_state_dict(
        model_zoo.load_url(
            "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
            map_location="cpu",
            check_hash=True,
        )
    )
    pre_gen.to(DEVICE)

    pre_results = evaluate_single_model(pre_gen, eval_loader, feature_extractor)
    results["pretrained_eccv"] = pre_results

    print("===== PRETRAINED ECCV-16 RESULTS =====")
    print(f"Avg L1 Loss:         {pre_results['avg_l1']:.4f}")
    print(f"Avg MSE Loss:        {pre_results['avg_mse']:.4f}")
    print(f"Avg Perceptual Loss: {pre_results['avg_perceptual']:.4f}")
    print(f"Raw AUC (0..{int(MAX_DISTANCE)}): {pre_results['auc_raw_norm']:.4f}")
    print(f"Avg PSNR:            {pre_results['avg_psnr']:.4f}")
    print(f"Avg SSIM:            {pre_results['avg_ssim']:.4f}")
    print(f"Total batches:       {pre_results['num_batches']}")
    print("======================================\n")

    return results


if __name__ == "__main__":
    results = evaluate_both_models(
        gan_weights_path="gan_generator_epoch_50.pth",
        split="validation",   
        batch_size=16,
        max_batches=None,    
    )
