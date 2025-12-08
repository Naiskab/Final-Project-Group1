# Code Overview and Execution Guide

This folder contains all code for the project.
The scripts cover three stages of the pipeline:

1. **ECCV16-style classification fine-tuning** on ImageNet-50 (313-bin color classification with rebalanced cross-entropy).  
2. **Adversarial fine-tuning with a PatchGAN** on top of the ECCV16 generator.  
3. **Quantitative evaluation and a Streamlit demo app** comparing the pretrained ECCV16 model against the fine-tuned GAN.

---

## Folder Structure

- `pretrained_finetune.py`  
  ECCV16 colorization model with 313-bin classification head, fine-tuned on ImageNet-50 using rebalanced cross-entropy.

- `gan_finetune.py`  
  GAN training script that uses the ECCV16 generator as the backbone and adds a PatchGAN discriminator and perceptual loss.

- `model_inference.py`  
  Evaluation script that runs both the fine-tuned GAN generator and the original ECCV16 pretrained model on the same split and reports metrics (PSNR, SSIM, and AuC).

- `streamlit_app.py`  
  Interactive Streamlit demo to upload a grayscale image and visualize three images side by side: ground truth, pretrained ECCV16 colorization, and fine-tuned GAN colorization.

---

## 1. Environment Setup

### Dependencies

Install dependencies in your environment:

```bash
pip install torch torchvision torchaudio \
            datasets \
            pillow \
            scikit-image \
            tqdm \
            matplotlib \
            pandas \
            streamlit
```

## 2. ECCV16 Fine-Tuning with Rebalanced Cross-Entropy

`pretrained_finetune.py`

### What the script does

1. Loads the ImageNet-50 subset from Hugging Face (Elriggs/imagenet-50-subset).

2. Downloads or loads pts_in_hull.npy (313 color cluster centers) and computes a mapping from continuous ab values to discrete bins.

3. Computes class-rebalancing weights over the training split.

4. Fine-tunes the ECCV16 model with rebalanced cross-entropy for EPOCHS epochs.

5. Saves qualitative examples during training in finetune_train/.

6. Saves the final fine-tuned model as:

`colorization_model_finetuned_ce.pth`

### How to run
`python3 pretrained_finetune.py`

## 3. GAN Fine-Tuning with PatchGAN and Perceptual Loss

`gan_finetune.py`

### What the script does

1. Loads the ImageNet-50 subset from Hugging Face (Elriggs/imagenet-50-subset, split='train').

2. Builds:

- Generator: ECCV16-style network operating in ab space.

- Discriminator: PatchGAN discriminator over concatenated (L, ab) patches.

- Feature extractor: VGG19 for perceptual loss.

3. Trains the GAN with:

- GAN loss (with label smoothing),

- L1 loss in ab space,

- Perceptual loss on LAB images.

4. Logs training losses to training_losses.csv and saves a sample image per epoch to gan_samples/.

5. Saves:

- Best generator: best_gan_generator.pth (saved at: [model link](https://drive.google.com/file/d/1c5hy7IHJ2E0C15fUdar_vN7IHKqc3_NE/view?usp=drive_link))

- Best discriminator: best_gan_discriminator.pth

### How to run

`python3 gan_finetune.py`

## Important Note

Model evaluation script and the Streamli app scripts load best_gan_discriminator.pth. Please refer to the link mentioned above to download the model file and make sure the model sits in the same directory to run the codes.

## 4. Quantitative Model Evaluation

`model_inference.py`

For each model, the script calculates:

- PSNR in RGB space ([0, 1])

- SSIM in RGB space ([0, 1])

- AuC of the cumulative distribution of per-pixel L2 errors in ab space, normalized over [0, 150] (following Zhang et al., 2016)

### How to run

`python3 model_inference.py`

## 5. Streamlit Demo Application

`streamlit_app.py`

### What the app does

After you upload an image, the app:

1. Runs two models:

- Pretrained ECCV16 generator (downloaded from the official URL).

- Fine-tuned GAN generator loaded from your checkpoint.

2. Displays three columns side by side:

- Original uploaded RGB image 

- Colorization from the pretrained ECCV16 model.

- Colorization from the fine-tuned GAN generator.

### How to run

`streamlit run streamlit_app.py`
