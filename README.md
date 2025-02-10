# GSVC: Efficient Video Representation and Compression Through 2D Gaussian Splatting
<!-- [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  -->
[![arXiv](https://img.shields.io/badge/GSVC-2501.12060-red
)](https://arxiv.org/abs/2501.12060)
[![GitHub Repo stars](https://img.shields.io/github/stars/LonganWANG-cs/GSVC?style=social&label=Star
)](https://github.com/LonganWANG-cs/GSVC)

[[paper](https://arxiv.org/abs/2501.12060)][[project page](https://yuang-ian.github.io/gsvc/)][[code](https://github.com/LonganWANG-cs/GSVC)]

[Longan Wang](https://longanwang-cs.github.io/), [Yuang Shi](https://yuang-ian.github.io/), [Wei Tsang Ooi📧](https://www.comp.nus.edu.sg/~ooiwt/)
(📧 denotes corresponding author.)


This is the official implementation of our paper [GSVC](https://arxiv.org/abs/2501.12060), an approach to learning a set of 2D Gaussian splats that can effectively represent and compress video frames.
GSVC incorporates the following techniques: 
(i) To exploit temporal redundancy among adjacent frames, which can speed up training and improve the compression efficiency, we predict the Gaussian splats of a frame based on its previous frame.
(ii) To control the trade-offs between file size and quality, we remove Gaussian splats with low contribution to the video quality; (iii) To capture dynamics in videos, we randomly add Gaussian splats to fit content with large motion or newly-appeared objects;
(iv) To handle significant changes in the scene, we detect key frames based on loss differences during the learning process.  
Experiment results show that GSVC achieves good rate-distortion trade-offs, comparable to state-of-the-art video codecs such as AV1 and HEVC, and a rendering speed of 1500 fps for a 1920×1080 video.
More qualitative results can be found in our paper.

<figure style="text-align: center;">
  <div>
    <img src="./img/q_UVG_Bpp_vs_PSNR.png" alt="UVG_PSNR" width="239" />
    <img src="./img/q_UVG_Bpp_vs_MS-SSIM.png" alt="UVG_MSSSIM" width="239" />
    <img src="./img/q_UVG_Bpp_vs_VMAF.png" alt="UVG_VMAF" width="320" />
  </div>
  <figcaption><strong>Figure 1:</strong> Rate-Distortion Curves in PSNR, MS-SSIM, and VMAF: comparison of our approach with baselines.</figcaption>
</figure>


## News

* **2025/2/1**: 🔥 We release our Python and CUDA code for GSVC presented in our paper. Have a try!

<!-- * **2025/2/1**: 🌟 Our paper has been accepted by NOSSDAV 2025! 🎉 Cheers! -->

## Insight
Figure 2 and Figure 3 illustrate how Gaussian splats can serve as a primitive for representing an image, and how the training process iteratively optimizes the parameters of the Gaussian splats to fit a given image.
<figure style="text-align: center;">
  <div>
    <img src="./img/Insight1.png" alt="Insight"/>
  </div>
  <figcaption><strong>Figure 2:</strong> Using Gaussian splats to model an image. Image is taken from UVG Jockey video. The sub-figures show that as the number of Gaussian splats N increases, the learned splats increasingly approximate the image’s content.</figcaption>
</figure>

<figure style="text-align: center;">
  <div>
    <img src="./img/Insight2.png" alt="Insight" />
  </div>
  <figcaption><strong>Figure 3:</strong> Image taken from UVG Jockey video. Intermediate training results after t iterations for 10,000 Gaussian splats, illustrating how Gaussian splats parameters are optimized to fit the content of a frame.</figcaption>
</figure>

## Overview

We propose a novel video representation and compression framework termed GSVC based on 2D Gaussian splats. Similar to modern video codecs, GSVC categorizes video frames into key-frames (Iframe) and predicted frames (P-frames).
Gaussian splats representation for the I-frame is learned from scratch while,
for P-frames, it is learned incrementally from its previous frame.
The predicted frames allow us to exploit temporal redundancy among the frames.
To allow rate control to trade off between compression rate and quality, GSVC prunes Gaussian splats with low contributions to the frame quality.
To cater to video dynamics, GSVC augments each Pframe with random splats before optimization. Finally, we monitor the loss values when learning the splats of each frame to determine if a frame should be an I-frame or a P-frame. 
This step allows GSVC to detect significant changes in the scene.


## Play with Our Code

### Cloning the Repository

```shell
# SSH
git clone git@github.com:LonganWANG-cs/GaussianVideo.git
```
or
```shell
# HTTPS
git clone https://github.com/LonganWANG-cs/GaussianVideo.git
```
After cloning the repository, you can follow these steps to train GSVC models under different tasks. 

### Requirements

We are using `python version 3.9` and `torch cuda runtime 12.4`. Before running the below code, please make sure `torch` is installed.

```bash
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

If you encounter errors while installing the packages listed in requirements.txt, you can try installing each Python package individually using the pip command.

Before training, you need to download the [UVG](https://ultravideo.fi/dataset.html). 

### Manual Training Example
This repository provides scripts for video representation and compression using Gaussian models. The workflow involves two key steps:

1. **Video Representation:** Overfit a video using Gaussian splatting to obtain a detailed model.
2. **Video Compression:** Compress the overfitted model to reduce file size while maintaining quality.

#### Step 1: Video Representation
Run `train_video_Represent.py` to generate the overfitted model:

```bash
python train_video_Represent.py \
  --loss_type L2 \
  --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
  --data_name Beauty \
  --num_points 10000 \
  --savdir GaussianVideo_results \
  --savdir_m GaussianVideo_models \
  --iterations 100000 \
  --is_rm --is_ad
```

#### Step 2: Video Compression
Run `train_video_Compress.py` to compress the video using the overfitted model:

```bash
python train_video_Compress.py \
  --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
  --model_path /home/e/e1344641/GaussianVideo/checkpoints/GaussianVideo_models/Beauty/GaussianVideo_100000_10000/gmodels_state_dict.pth \
  --data_name Beauty \
  --num_points 10000 \
  --savdir GaussianVideo_results \
  --savdir_m GaussianVideo_models \
  --iterations 50000 \
  --is_rm
```

### SLURM Job Scripts

For automated training and compression on SLURM, `.sh` files are provided:

- **Complete Workflow:** Run `sh_train_compression.sh` to perform both representation and compression.

  ```bash
  sbatch sh_train_compression.sh
  ```

- **Representation Only:** Run `sh_train_representation.sh` if you only want to train the model without compression.

  ```bash
  sbatch sh_train_representation.sh
  ```








## Acknowledgments

Our code was developed based on [GaussianImage](https://github.com/Xinjie-Q/GaussianImage). 
This is a paradigm of image representation and compression by 2D Gaussian Splatting.


## Citation

If you find our GSVC useful or relevant to your research, please kindly cite our paper:

```
@article{wang2025gsvc,
  title={GSVC: Efficient Video Representation and Compression Through 2D Gaussian Splatting},
  author={Wang, Longan and Shi, Yuang and Ooi, Wei Tsang},
  journal={arXiv preprint arXiv:2501.12060},
  year={2025}
}
```
