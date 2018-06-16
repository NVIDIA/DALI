# DALI

DALI is a collection of highly optimized building blocks and an execution engine to accelerate computer vision deep learning applications. The goal of DALI is to provide both performance and flexibility, as a single library, that can be easily integrated into DL training and inference applications.

## Key features

- Full data pipeline accelerated from reading images on disk to getting ready for training/inference
- Flexibility through configurable graphs
- Support for image classification and segmentation workloads
- Ease of integration through direct framework plugins
- Portable training workflows with multiple input formats - JPEG images, LMDB, RecordIO, TFRecord

# Building and installing

## Prerequisities

- NVIDIA CUDA 9.0 or above

## Binaries

TODO

## From source

### Install prerequisities



### Get the DALI source

```
git clone --recursive https://github.com/NVIDIA/dali
cd dali
```

### Make the build directory

```
mkdir build
cd build
```

### Compile DALI

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

## Docker image

Dockerfile is supplied. To build:

```
docker build -t dali -f Dockerfile
```

# Getting started

[`examples`](examples) directory contains a series of examples (in the form of Jupyter notebooks) of different features of DALI. It also contains examples of how to use DALI to interface with DL frameworks.
