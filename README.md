[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# NVIDIA DALI v0.1

Today’s deep learning applications include complex, multi-stage pre-processing data pipelines that include compute-intensive steps mainly carried out on the CPU. For instance, steps such as load data from disk, decode, crop, random resize, color and spatial augmentations and format conversions are carried out on the CPUs, limiting the performance and scalability of training and inference tasks. In addition, the deep learning frameworks today have multiple data pre-processing implementations, resulting in challenges such as portability of training and inference workflows and code maintainability.

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks and an execution engine to accelerate input data pre-processing for deep learning applications. DALI  provides both performance and flexibility of accelerating different data pipelines, as a single library, that can be easily integrated into different deep learning training and inference applications.

Key highlights of DALI include:
 - Full data pipeline accelerated from reading disk to getting ready for training/inference
 - Flexibility through configurable graphs and custom operators
 - Support for image classification and segmentation workloads
 - Ease of integration through direct framework plugins and open source bindings
 - Portable training workflows with multiple input formats - JPEG, LMDB, RecordIO, TFRecord
 - Extensible for user specific needs through open source license

Note: DALI v0.1 is a pre-release software, which means certain features may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, accessibility, availability, and reliability standards relative to production-quality versions of NVIDIA software and materials. You may use a pre-release software at your own risk, understanding that pre-release software is not intended for use in production or business-critical systems.

# Installing prebuilt DALI packages

## Prerequisities

* Linux x64
* [NVIDIA Driver](https://www.nvidia.com/drivers) supporting [CUDA 9.0](https://developer.nvidia.com/cuda-downloads) or later
  - This corresponds to 384.xx and later driver releases.
* DALI can work with any of the following Deep Learning frameworks:
  - [MXNet](http://mxnet.incubator.apache.org)
    - Version 1.3 beta is required, `mxnet-cu90==1.3.0b20180612` or later
  - [pyTorch](https://pytorch.org)
    - Version 0.4
  - [TensorFlow](https://www.tensorflow.org)
    - Version 1.8

## Installation

`pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali`

# Compiling DALI from source

## Prerequisities

* Linux
* [NVIDIA CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
* [nvJPEG library](https://developer.nvidia.com/nvjpeg)
* [protobuf](https://github.com/google/protobuf) version 2 or above (version 3 or above is required for TensorFlow TFRecord file format support)
* [CMake](https://cmake.org) version 3.5 or above
* [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) version 1.5.x or above
* [OpenCV](https://opencv.org) version 3 or above
* (Optional) [liblmdb](https://github.com/LMDB/lmdb) version 0.9.x or above
* DALI can work with any of the following Deep Learning frameworks:
  - [MXNet](http://mxnet.incubator.apache.org)
    - Version 1.3 beta is required, `mxnet-cu90==1.3.0b20180612` or later
  - [pyTorch](https://pytorch.org)
    - Version 0.4
  - [TensorFlow](https://www.tensorflow.org)
    - Version 1.8
    - Note: Installing TensorFlow is required to build the TensorFlow plugin for DALI


## Get the DALI source

```
git clone --recursive https://github.com/NVIDIA/dali
cd dali
```

## Make the build directory

```
mkdir build
cd build
```

## Compile DALI

To build DALI without LMDB support:

```
cmake ..
make -j"$(nproc)" install
```

To build DALI with LMDB support:

```
cmake -DBUILD_LMDB=ON ..
make -j"$(nproc)" install
```

Optional CMake build parameters:

- `BUILD_PYTHON` - build Python bindings (default: ON)
- `BUILD_TEST` - include building test suite (default: ON)
- `BUILD_BENCHMARK` - include building benchmarks (default: ON)
- `BUILD_LMDB` - build with support for LMDB (default: OFF)
- `BUILD_NVTX` - build with NVTX profiling enabled (default: OFF)
- `BUILD_TENSORFLOW` - build TensorFlow plugin (default: OFF)

## Install Python bindings

```
pip install dali/python
```

# Getting started

[`examples`](examples) directory contains a series of examples (in the form of Jupyter notebooks) of different features of DALI. It also contains examples of how to use DALI to interface with DL frameworks.

# Contributing to DALI

Contributions to DALI are more than welcome.  To make the pull request process smooth, please follow these [`guidelines`](CONTRIBUTING).

# Contributors

DALI was built with major contributions from Trevor Gale, Przemek Tredak, Simon Layton, Andrei Ivanov, Serge Panev
