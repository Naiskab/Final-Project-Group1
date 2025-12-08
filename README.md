# Final-Project-Group1: Colorization of grayscale images using Deep Neural Networks

This repository contains the implementation and evaluation of grayscale image colorization using deep learning. The project extends the ECCV-2016 colorization model by integrating adversarial learning and perceptual loss, with the goal of improving perceptual realism in generated color outputs. Experiments compare a pretrained model against a fine-tuned adversarial model.

## Course Information
DATS 6312 – Deep Learning
The George Washington University

## Contributors
- Naiska Buyandalai
- Sneha Tadapaneni
- Sairachana Kandikattu

## Project Summary
This project investigates whether GAN-based fine-tuning improves perceptual quality in image colorization tasks. Beginning with the ECCV-2016 colorization architecture, we extend the method using adversarial discriminators and VGG-based perceptual loss. We evaluate results using multiple metrics (AUC, PSNR, SSIM, perceptual loss) and visual inspection.

## Important Note
Instructions for running, training, and evaluating the models are documented separately inside:
Code/README.md

## Methods Implemented
- ECCV-16 encoder–decoder colorization network
- Adversarial fine-tuning 
- Perceptual VGG loss
- L1 reconstruction loss
- Combined adversarial and perceptual optimization

## Model Inference and Demo
A Streamlit application is included for demonstration. The app performs inference using both:

1. the original ECCV-16 pretrained model
2. the fine-tuned GAN

Users may upload grayscale images and view side-by-side results at: https://dl-group1-colorization-app.streamlit.app/

