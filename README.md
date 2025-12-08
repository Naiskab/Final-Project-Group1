# Final-Project-Group1: Colorization of grayscale images using Deep Neural Networks

This repository contains the implementation and evaluation of grayscale image colorization using deep learning. The project extends the ECCV-2016 colorization model by integrating adversarial learning and perceptual loss, with the goal of improving perceptual realism in generated color outputs. Experiments compare a pretrained model against a fine-tuned adversarial model.

## Course Information
DATS 6312 – Deep Learning
The George Washington University

## Contributors
- Naiska Buyandalai
- Sneha Tadapaneni
- Sairachana Kandikattu

## Approach and Implementation

This project builds on the seminal ECCV-2016 encoder–decoder colorization network, initially trained with a multinomial classification objective over quantized ab channels. We extended this baseline by introducing adversarial learning with a PatchGAN discriminator and incorporating perceptual supervision through VGG feature loss. The hypothesis behind this extension is that pixel-aligned reconstruction alone often produces desaturated or low-contrast colors, while adversarial and perceptual losses encourage spatial coherence, richer colors, and more realistic textures.

Two complementary training pipelines were implemented:

1. ECCV pretrained (Zhang et al.) – used as a baseline reference

2. Fine-tuned GAN model – trained adversarially using HF ImageNet-50

Results Summary

Evaluations were made by multiple metrics, including AUC error in ab-space, PSNR and SSIM in RGB reconstruction space. Across validation testing, adversarially fine-tuned models produced subtle numerical improvements while achieving noticeably stronger perceptual quality.

Qualitative visualization demonstrates more vivid color saturation and reduced grayish artifacts.





## Important Note
Instructions for running, training, and evaluating the models are documented separately inside:
Code/README.md

## Methods Implemented
- ECCV-16 encoder–decoder colorization network
- Adversarial fine-tuning 
- Perceptual VGG loss
- L1 reconstruction loss
- Combined adversarial and perceptual optimization

## Demo
A Streamlit application is included for demonstration. The app performs inference using both:

1. the original ECCV-16 pretrained model
2. the fine-tuned GAN

Users may upload grayscale images and view side-by-side results at: https://dl-group1-colorization-app.streamlit.app/

