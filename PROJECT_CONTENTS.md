# Project Contents Breakdown

Title: `Generative AI for image synthesis using GANs`

Use `GANs`, not `GAN's`, in formal writing unless you mean possession.

## Major Contents To Include

1. Introduction
   - What generative AI means.
   - Why image synthesis is important.
   - Difference between real images and AI-generated images.
   - Project objectives: generation and real/fake classification.

2. Literature Background
   - Generative Adversarial Networks.
   - Generator and discriminator roles.
   - Loss functions and adversarial training.
   - Common problems: mode collapse, unstable training, blurry outputs.
   - Other synthesis methods such as VAE and diffusion models.

3. Dataset Description
   - 60k total images.
   - 30k real images and 30k fake or AI-generated images.
   - Mixed image sizes: mostly 256x256, some fake images at 128x128.
   - Preprocessing requirement: convert all images to the same RGB square size.

4. Proposed System
   - Dataset preprocessing.
   - GAN image synthesis module.
   - VAE image synthesis module as an alternative method.
   - Real/fake image classifier.
   - Streamlit interface for generation and classification.

5. Methodology
   - Resize and split data.
   - Train classifier on real and fake images.
   - Train GAN on real images.
   - Train VAE on real images.
   - Use trained checkpoints for inference in the app.

6. Implementation
   - Python.
   - PyTorch and Torchvision.
   - Pillow for image preprocessing.
   - Streamlit for the web page.
   - CPU-friendly settings for a no-GPU laptop.

7. Results And Evaluation
   - Classifier accuracy, validation loss, confusion matrix if added later.
   - Generated image sample grids across epochs.
   - Comparison between GAN and VAE outputs.
   - Observations on CPU training limitations.

8. Limitations
   - CPU-only training is slow.
   - DCGAN is practical but less powerful than large StyleGAN or diffusion models.
   - Classifier performance depends heavily on dataset diversity.
   - Generated images improve with more epochs and better hardware.

9. Future Scope
   - GPU training.
   - StyleGAN2-ADA or diffusion model fine-tuning.
   - Stronger classifier using pretrained EfficientNet or ViT.
   - Explainability with Grad-CAM.
   - Web deployment.

10. Ethics
   - AI-generated images can be misused.
   - Detection systems are probabilistic, not absolute truth.
   - Dataset privacy and responsible use should be documented.
