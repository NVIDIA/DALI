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

# Installing prebuilt DALI packages

## Prerequisities

* **Linux x64**
* **[NVIDIA Driver](https://www.nvidia.com/drivers)** supporting [CUDA 9.0](https://developer.nvidia.com/cuda-downloads) or later (i.e., 384.xx or later driver releases)
* One or more of the following Deep Learning frameworks:
  - **[MXNet 1.3 beta](http://mxnet.incubator.apache.org)** `mxnet-cu90==1.3.0b20180612` or later
  - **[pyTorch 0.4](https://pytorch.org)**
  - **[TensorFlow 1.7](https://www.tensorflow.org)** or later

## Installation

`pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali`

# Compiling DALI from source

## Prerequisities

* **Linux x64**
* **[NVIDIA CUDA 9.0](https://developer.nvidia.com/cuda-downloads)**
  *(CUDA 8.0 compatibility is provided unofficially)*
* **[nvJPEG library](https://developer.nvidia.com/nvjpeg)**<br/>
  *(This can be unofficially disabled; see below)*
* **[protobuf](https://github.com/google/protobuf)** version 2 or later (version 3 or later is required for TensorFlow TFRecord file format support)
* **[CMake 3.5](https://cmake.org)** or later
* **[libjpeg-turbo 1.5.x](https://github.com/libjpeg-turbo/libjpeg-turbo)** or later<br/>
  *(This can be unofficially disabled; see below)*
* **[OpenCV 3](https://opencv.org)** or later
  *(OpenCV 2.x compatibility is provided unofficially)*
* **(Optional) [liblmdb 0.9.x](https://github.com/LMDB/lmdb)** or later
* One or more of the following Deep Learning frameworks:
  - **[MXNet 1.3 beta](http://mxnet.incubator.apache.org)** `mxnet-cu90==1.3.0b20180612` or later
  - **[pyTorch 0.4](https://pytorch.org)**
  - **[TensorFlow 1.7](https://www.tensorflow.org)** or later<br/>
  *Note: TensorFlow installation is required to build the TensorFlow plugin for DALI*

> NOTE: Items marked *"unofficial"* are community contributions that are
> believed to work but not officially tested or maintained by NVIDIA.

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
- *[Unofficial]* `BUILD_JPEG_TURBO` - build with libjpeg-turbo (default: ON)
- *[Unofficial]* `BUILD_NVJPEG` - build with nvJPEG (default: ON)

## Install Python bindings

```
pip install dali/python
```

# Getting started

[`docs/examples`](docs/examples) directory contains a series of examples (in the form of Jupyter notebooks) of different features of DALI. It also contains examples of how to use DALI to interface with DL frameworks.

# Contributing to DALI

Contributions to DALI are more than welcome.  To make the pull request process smooth, please follow these [`guidelines`](CONTRIBUTING).

# Contributors

DALI was built with major contributions from Trevor Gale, Przemek Tredak, Simon Layton, Andrei Ivanov, Serge Panev
