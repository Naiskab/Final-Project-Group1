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
    BATCH_SIZE)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_DISTANCE = 150.0
N_BINS = 151

def evaluate_generator(
    gen_weights_path="gan_generator_epoch_50.pth",
    data_folder="../imagenet_50/validation",
    batch_size=None,
    save_examples=True,
    examples_dir="gen_eval_examples",
    max_batches=None,
):
    """
    Evaluates the model.
    Computes:
        - Raw AuC (consistent to Zhang et al paper): area under cumulative distribution of per-pixel L2 color error in ab space, 0..150
        - PSNR (RGB space, [0,1])
        - SSIM (RGB space, [0,1])
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    # -------------------------------------------------------
    # Load generator
    # -------------------------------------------------------
    # ---------------------------------------
    # Finetuned GAN generator
    # ---------------------------------------
    print(f"Loading finetuned GAN generator: {gen_weights_path}")
    generator = ECCVGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(gen_weights_path, map_location=DEVICE))
    generator.eval()

    # ---------------------------------------
    # Pretrained ECCV model
    # ---------------------------------------
    # print("Loading pretrained ECCV model from Zhang et al. (2016)")
    # generator = ECCVGenerator()
    # generator.load_state_dict(
    #     model_zoo.load_url(
    #         'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
    #         map_location='cpu',
    #         check_hash=True
    #     )
    # )
    # generator.to(DEVICE)
    # generator.eval()

    # ------------------------------------------------------
    # Feature extractor (for perceptual loss)
    # ------------------------------------------------------
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

    # generate sample outputs
    if save_examples:
        os.makedirs(examples_dir, exist_ok=True)

        # select first 5 images per label
        labels_path = os.path.join(os.path.dirname(data_folder), "labels.csv")
        labels_df = pd.read_csv(labels_path)

        val_filenames = set(os.listdir(data_folder))
        labels_df = labels_df[labels_df["filename"].isin(val_filenames)]

        # sort by image_id, then take first 5 per label
        labels_df = labels_df.sort_values("image_id")
        selected_rows = labels_df.groupby("label").head(5)
        filename_to_label = dict(zip(selected_rows["filename"], selected_rows["label"]))
        selected_filenames = set(filename_to_label.keys())
    else:
        selected_filenames = set()

    # -------------------------------------------------------
    # Accumulate losses and error for AuC
    # -------------------------------------------------------
    total_l1 = 0.0
    total_mse = 0.0
    total_perc = 0.0
    n_batches = 0

    total_psnr = 0.0
    total_psnr_count = 0

    total_ssim = 0.0
    total_ssim_count = 0

    # Histogram over error distances
    error_hist = torch.zeros(N_BINS, dtype=torch.float64)
    total_pixels = 0

    with torch.no_grad():
        for batch_idx, (tens_l, tens_ab_gt) in enumerate(tqdm(eval_loader)):
            tens_l = tens_l.to(DEVICE)
            tens_ab_gt = tens_ab_gt.to(DEVICE)

            pred_ab = generator(tens_l)

            # -----------------------
            # Compute L1, MSE, Perceptual
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

            # -----------------------
            # PSNR & SSIM in RGB space (values in [0,1])
            # -----------------------
            for i in range(tens_l.size(0)):
                gray_cpu = tens_l[i].cpu()
                real_ab_cpu = tens_ab_gt[i].cpu()
                fake_ab_cpu = pred_ab[i].cpu()

                # Convert to RGB in [0,1]
                real_rgb = postprocess_tens(gray_cpu, real_ab_cpu)
                fake_rgb = postprocess_tens(gray_cpu, fake_ab_cpu)

                # PSNR
                mse_rgb = np.mean((fake_rgb - real_rgb) ** 2)
                if mse_rgb > 0:
                    psnr_val = 10.0 * np.log10(1.0 / mse_rgb)  # MAX=1.0
                else:
                    psnr_val = float('inf')

                total_psnr += psnr_val
                total_psnr_count += 1

                # SSIM
                # skimage expects HWC
                ssim_val = ssim(
                    real_rgb,
                    fake_rgb,
                    data_range=1.0,
                    channel_axis=-1
                )
                total_ssim += ssim_val
                total_ssim_count += 1

            n_batches += 1

            # -----------------------
            # Update AuC histogram
            # -----------------------
            diff = pred_ab - tens_ab_gt
            err = torch.sqrt(torch.sum(diff ** 2, dim=1))
            err_flat = err.reshape(-1).cpu()
            err_flat = torch.clamp(err_flat, 0.0, MAX_DISTANCE)

            batch_hist = torch.histc(
                err_flat,
                bins=N_BINS,
                min=0.0,
                max=MAX_DISTANCE
            )

            error_hist += batch_hist
            total_pixels += err_flat.numel()

            # -----------------------
            # Save selected examples (first 5 per label)
            # -----------------------
            if save_examples:
                batch_size_eff = tens_l.size(0)
                for i in range(batch_size_eff):
                    # global index into eval_dataset
                    dataset_idx = batch_idx * batch_size + i
                    if dataset_idx >= len(eval_dataset):
                        continue

                    img_path = eval_dataset.image_paths[dataset_idx]
                    fname = os.path.basename(img_path)

                    # only save if this filename is one of the selected ones
                    if fname not in selected_filenames:
                        continue

                    gray = tens_l[i].cpu()
                    real_ab = tens_ab_gt[i].cpu()
                    fake_ab = pred_ab[i].cpu()

                    gray_rgb = postprocess_tens(gray, torch.zeros_like(real_ab))
                    real_rgb = postprocess_tens(gray, real_ab)
                    fake_rgb = postprocess_tens(gray, fake_ab)

                    comparison = np.concatenate([gray_rgb, real_rgb, fake_rgb], axis=1)

                    base, _ = os.path.splitext(fname)
                    label = filename_to_label.get(fname, "NA")
                    out_name = f"{base}_label{label}_comparison.png"
                    Image.fromarray((comparison * 255).astype(np.uint8)).save(
                        os.path.join(examples_dir, out_name)
                    )

            if max_batches is not None and n_batches >= max_batches:
                break

    # -------------------------------------------------------
    # Final scalar losses
    # -------------------------------------------------------
    avg_l1 = total_l1 / n_batches
    avg_mse = total_mse / n_batches
    avg_perc = total_perc / n_batches

    if total_psnr_count > 0:
        avg_psnr = total_psnr / total_psnr_count
    else:
        avg_psnr = float('nan')

    if total_ssim_count > 0:          # >>> SSIM
        avg_ssim = total_ssim / total_ssim_count
    else:
        avg_ssim = float('nan')

    # -------------------------------------------------------
    # Compute AuC from histogram
    # -------------------------------------------------------
    if total_pixels > 0:
        pdf = error_hist / float(total_pixels)
        cmf = torch.cumsum(pdf, dim=0)

        thresholds = torch.linspace(0.0, MAX_DISTANCE, steps=N_BINS)
        auc_raw = torch.trapz(cmf, thresholds).item()
        auc_norm = auc_raw / MAX_DISTANCE
    else:
        auc_norm = float('nan')

    print("\n===== GENERATOR EVALUATION RESULTS =====")
    print(f"Avg L1 Loss:         {avg_l1:.4f}")
    print(f"Avg MSE Loss:        {avg_mse:.4f}")
    print(f"Avg Perceptual Loss: {avg_perc:.4f}")
    print(f"Raw AuC (0..{int(MAX_DISTANCE)}): {auc_norm:.4f}")
    print(f"Avg PSNR:            {avg_psnr:.4f}")
    print(f"Avg SSIM:            {avg_ssim:.4f}")
    print(f"Total batches:       {n_batches}")
    print("========================================\n")

    return {
        "avg_l1": avg_l1,
        "avg_mse": avg_mse,
        "avg_perceptual": avg_perc,
        "auc_raw_norm": auc_norm,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "num_batches": n_batches,
    }


if __name__ == "__main__":
    results = evaluate_generator(
        gen_weights_path="gan_generator_epoch_50.pth",
        data_folder="../imagenet_50/validation",
        batch_size=16,
        save_examples=True,
        examples_dir="gen_eval_examples",
        max_batches=None,
    )
