# DALI Pipeline ZOO

## Prerequisites

To run the examples in this repository, you need:

- Python >= 3.8
- CUDA > 11
- NVIDIA DALI

### Installing Python

Visit the [official Python website](https://www.python.org/) for installation instructions.

### Installing CUDA

For CUDA installation, refer to the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### Installing NVIDIA DALI

To install NVIDIA DALI with CUDA 12.0 support, run:

```
pip install nvidia-dali-cuda120
```

## Overview

This repository contains a collection of examples and code snippets demonstrating various NVIDIA DALI pipelines. DALI (Data Loading Library) is a library for data loading and pre-processing to accelerate deep learning applications.

## Usage

The code snippets provided in this repository are designed to be used right out of the box. You can run them as-is or modify them to suit your specific needs and integrate them into your own projects. Each example is structured to help you understand how to leverage DALI for different data processing tasks efficiently.

## Repository Content

The repository is organized into the following folders:

### images

This folder contains DALI pipelines for processing image data. These pipelines demonstrate tasks such as image decoding, augmentation, and preprocessing for computer vision tasks.

### videos

Here you'll find DALI pipelines specifically designed for video processing. These examples showcase video decoding, frame extraction, and other video-related operations.

## Contributing

We welcome contributions to this repository. To contribute to DALI Pipeline ZOO and make pull requests, follow the guidelines outlined in [DALI Contribution Guide](https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md).
