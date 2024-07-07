# TiNO-Edit: Timestep and Noise Optimization for Robust Diffusion-Based Image Editing (CVPR 2024)

[[Arxiv](https://arxiv.org/abs/2404.11120)] [[Poster](https://cvpr.thecvf.com/virtual/2024/poster/31387)] [[Youtube](https://www.youtube.com/watch?v=latSSiMcfds)]

TiNO-Edit is an image editing algorithm built on top of Stable Diffusion (SD) by optimizing noise and timesteps in the SD latent space.

## Capabilities

<p align="center">
  <img src="https://github.com/SherryXTChen/TiNO-Edit/blob/main/assets/teaser.png" alt="Figure 1" width="32%">
  <img src="https://github.com/SherryXTChen/TiNO-Edit/blob/main/assets/compounded_image_editing.png" alt="Figure 2" width="46%">
</p>

## LatentCLIPvis & LatentVGG

LatentCLIPvis and LatentVGG are the SD latent space equivalents of the CLIP vision model and VGG. To train these models, see `latentclip` and `latentvgg` respectively. We will provide an explanation of our code soon. We also provide pretrained checkpoints [here](https://www.dropbox.com/scl/fo/0jdk7kddwtfstpc0gshx4/h?rlkey=wdfejfkf5sho513v8l7ddw2px&st=e3nhfopc&dl=0).

## Method (Code coming soon ...)

<p align="center">
  <img src="https://github.com/SherryXTChen/TiNO-Edit/blob/main/assets/method_overview.png" alt="Method overview" width="80%">
</p>

<p align="center">
  <img src="https://github.com/SherryXTChen/TiNO-Edit/blob/main/assets/pseudocode.png" alt="Pseudocode" width="80%">
</p>



``` bibtex
@inproceedings{chen2024tino,
  title={TiNO-Edit: Timestep and Noise Optimization for Robust Diffusion-Based Image Editing},
  author={Chen, Sherry X and Vaxman, Yaron and Ben Baruch, Elad and Asulin, David and Moreshet, Aviad and Lien, Kuo-Chin and Sra, Misha and Sen, Pradeep},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6337--6346},
  year={2024}
}
```
