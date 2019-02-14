# Example NVVL usage: Video Super-Resolution

In this example we use NVVL (**NV**idia **V**ideo **L**oader) to supply data for training a video super-resolution network implemented in [PyTorch](https://github.com/pytorch/pytorch).  We replicate the network described in [End-to-End Learning of Video Super-Resolution with Motion Compensation](https://arxiv.org/abs/1707.00471) Fig. 1(b).

The network implemented, VSRNet, uses an odd number of successive frames randomly sampled from a video as input.  The outer frames are warped to align with the center frame using a pretrained optical flow network FlowNetSD [2]. The warped frames are then passed to a convolutional network that predicts a single higher resolution version of the center frame.

We make a number of small modifications to the network that we found improved the convergence
rate, e.g. batch normalization, Adam optimizer, cyclic learning rate policy.

Single node, multi-GPU training code is provided (single GPU is not supported).  fp32 and fp16 training are supported.

## Dataloaders

Two data loader options are provided for comparison:

- NVVL allows random access to frame sequences directly from .mp4 files
- A standard PyTorch dataloader loads frames from individual .png files

These dataloaders can be found in [dataloading/dataloaders.py](./dataloading/dataloaders.py).

## Data loader performance


We present performance characteristics of NVVL and a standard .png data loader
for a number of VSRNet training scenarios. In all cases we find that NVVL offers
reduced CPU load, host memory usage, disk usage and per iteration data loading
time.

| Data loader | Input resolution | Crop size\* | Numerical precision | PyTorch loader Workers (.png only) | Batch size | CPU load\*\* | Peak host memory (%) | Disk space for dataset(GB) | Per iteration data time (ms)\*\*\* |
| ---- | ----- | ---- | ---- | --- | - | -- | ---- | ----- | ---- |
| NVVL | 540p  | None | fp32 | N/A | 7 | 10 | 4.0  | 0.592 | 0.91 |
| .png | 540p  | None | fp32 | 10  | 7 | 17 | 4.7  | 23    | 2.73 |
| NVVL | 540p  | None | fp16 | N/A | 7 | 10 | 4.0  | 0.592 | 0.21 |
| .png | 540p  | None | fp16 | 10  | 7 | 19 | 4.7  | 23    | 3.15 |
| NVVL | 720p  | 540p | fp32 | N/A | 4 | 10 | 4.0  | 0.961 | 0.28 |
| .png | 720p  | 540p | fp32 | 10  | 4 | 18 | 4.7  | 38    | 2.66 |
| NVVL | 720p  | 540p | fp16 | N/A | 4 | 10 | 4.0  | 0.961 | 0.35 |
| .png | 720p  | 540p | fp16 | 10  | 4 | 20 | 4.8  | 38    | 2.66 |

\* Random cropping

\*\* [CPU load](https://en.wikipedia.org/wiki/Load_(computing)) steady state 1 min
  average

\*\*\* NOTE: We must insert cuda synchronization to measure the average time to load
a batch.  The table shows that the average time for loading from .png is much
higher than loading from .mp4 using nvvl. However, due to asynchronous data loading this latency will not be seen in the total per iteration time unless the computation performed during an iteration takes less than the data loading time.

## Requirements

The system used to run this project must include two or more Kepler, Pascal, Maxwell or Volta NVIDIA GPUs.

Software requirements:

* Cuda 9.0+
* Pytorch 0.4+
* ffmpeg=3.4.2
* scikit-image
* tensorflow
* tensorboard
* tensorboardX

## Installation

We recommend using Docker for building an environment with the appropriate dependencies for this project.

A Dockerfile is provided at [./docker/Dockerfile](./docker/Dockerfile). The base
container it inherets from is available by signing up for [NVIDIA GPU Cloud
(NGC)](https://ngc.nvidia.com/). Alternatively you can build a PyTorch container from top-of-tree source
following the instructions [here](https://github.com/pytorch/pytorch#docker-image) and inheret from that container instead.  To
build, run the following from the root directory of this repo:

    cd examples/pytorch_superres/docker
    docker build -t vsrnet .

## FlowNet2-SD implementation and pre-trained model

We make use of the FlowNet2-SD PyTorch implementation available [here](https://github.com/NVIDIA/flownet2-pytorch).  It is included in this repo as a git submodule.

In order to use the pre-trained FlowNet2-SD network run the following from the
root directory of this repo:

    git submodule init
    git submodule update

Training the VSRNet implemented here requires the use of pre-trained weights from the FlowNet2-SD network.  We provided a converted Caffe pre-trained model below.  Should you use these weights, please adhere to the [license agreement](https://drive.google.com/file/d/1TVv0BnNFh3rpHZvD-easMb9jYrPE2Eqd/view?usp=sharing):

[FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]

The default location that the training code will look for these weights is `examples/pytorch_superres/flownet2-pytorch/networks/FlowNet2-SD_checkpoint.pth.tar`. This location can be changed via the `--flownet_path` argument to `main.py`.

## Data

Following [Makansi et al.](https://arxiv.org/abs/1707.00471) we use the [Myanmar
60p video](https://www.harmonicinc.com/resources/videos/4k-video-clip-center) as our
raw data source.

The raw video is a 60 FPS, 4K resolution cinematic video.  In order to prepare
the data for training you should run the following steps:

0. Create a data folder `<data_dir>` and download the 4K Myanmar video there.

1. Split the video into scenes and remove audio track:

```bash
nvidia-docker run --rm -it --ipc=host --net=host -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -v $PWD:/workspace -v <data_dir>:<data_dir> -u $(id -u):$(id -g) vsrnet /bin/bash

python ./tools/split_scenes.py --raw_data <path_to_mp4_file> --out_data <data_dir>
```

The scenes will be written to `<data_dir>/orig/scenes`.  The scenes will
be split into training and validation folders.

2. Transcode the scenes to have a smaller keyframe interval and possibly a lower resolution:

```bash
nvidia-docker run --rm -it --ipc=host --net=host -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -v $PWD:/workspace -v <data_dir>:<data_dir> -u $(id -u):$(id -g) vsrnet /bin/bash

python ./tools/transcode_scenes.py --master_data <data_dir> --resolution <resolution>
```

where `<resolution>` can be one of: '4K', `1080p`, `720p` or `540p`.  The transcoded scenes will be written to `<data_dir>/<resolution>/scenes` and split into
training and validation folders. Run the script with `--help` to see more options. Note that while you can split and transcode the original video in one step, we found it to be much faster to split first, then transcode.

3. Extract .png frames from scene .mp4 files:

```bash
nvidia-docker run --rm -it --ipc=host --net=host -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -v $PWD:/workspace -v <data_dir>:<data_dir> -u $(id -u):$(id -g) vsrnet /bin/bash

python ./tools/extract_frames.py --master_data <data_dir> --resolution <resolution>
```

where `<resolution>` can be one of: `1080p`, `720p` or `540p`.  The extracted frames will be written to `<data_dir>/<resolution>/frames/` and split into
training and validation folders which will be split into one folder per scene.

## Training

Training can be run by running the following command:

    bash run_docker_distributed.sh

This will initialize a PyTorch distributed dataparallel training run using all
GPUs available in the host.  This file allows configuration of a variety of
training options - it is expected that you will modify data paths appropriately
for your system.

Visualization of training data, e.g. loss curves and timings, aswell as sample images is provided through [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) via the [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) library.  Whilst training is running you can access Tensorboard at `<host_ip>:6006`.

## Results on Myanmar validation set

All testing of this project was carried out on an NVIDIA DGX-1 using all 8 V100 GPUs and running CUDA 9.1, PyTorch 0.4.0a0+02b758f, cuDNN v7.0.5 in Ubuntu 16.04 Docker containers.

Input image (128x240 - click to see actual size):

![](./data/input.png)

VSRNet prediction (512x960 - click to see actual size):

![](./data/predicted.png)

Example training loss (fp16, batch size 7, min_lr=max_lr=0.001):

![](./data/train_loss.png)

Example validation PSNR (fp16, batch size 7, min_lr=max_lr=0.001)

![](./data/val_psnr.png)

## Reference
If you find this implementation useful in your work, please acknowledge it appropriately and cite the following papers:
````
@InProceedings{IB17,
  author       = "O. Makansi and E. Ilg and and Thomas Brox",
  title        = "End-to-End Learning of Video Super-Resolution with Motion Compensation",
  booktitle    = "German Conference on Pattern Recognition (GCPR) 2017",
  month        = " ",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2017/IB17"
}
````

````
@InProceedings{IMKDB17,
  author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month        = "Jul",
  year         = "2017",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
````

