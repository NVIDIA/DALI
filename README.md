[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Documentation](https://img.shields.io/badge/Nvidia%20DALI-documentation-brightgreen.svg?longCache=true)](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/)


# NVIDIA DALI

Deep learning applications require complex, multi-stage pre-processing data pipelines. Such data pipelines involve compute-intensive operations that are carried out on the CPU. For example, tasks such as: load data from disk, decode, crop, random resize, color and spatial augmentations and format conversions, are mainly carried out on the CPUs, limiting the performance and scalability of training and inference.

In addition, the deep learning frameworks have multiple data pre-processing implementations, resulting in challenges such as portability of training and inference workflows, and code maintainability.

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks, and an execution engine, to accelerate the pre-processing of the input data for deep learning applications. DALI  provides both the performance and the flexibility for accelerating different data pipelines as a single library. This single library can then be easily integrated into different deep learning training and inference applications.

## Highlights


* Full data pipeline--accelerated from reading the disk to getting ready for training and inference.
* Flexibility through configurable graphs and custom operators.
* Support for image classification and segmentation workloads.
* Ease of integration through direct framework plugins and open source bindings.
* Portable training workflows with multiple input formats--JPEG, PNG (fallback to CPU), TIFF (fallback to CPU), BMP (fallback to CPU), raw formats, LMDB, RecordIO, TFRecord.
* Extensible for user-specific needs through open source license.


----

## Table of Contents


- [Installation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/quickstart.html)
- [Getting started](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html)
- [Examples and Tutorials](#examples)
- [Additional resources](#additional-resources)
- [Contributing to DALI](#contributing-to-dali)
- [Reporting problems, asking questions](#reporting-problems)
- [Contributors](#contributors)
    
----

## Examples

The [docs/examples](https://github.com/rajkaramchedu-nvidia/DALI/blob/master/docs/examples) directory contains a few examples (in the form of Jupyter notebooks) highlighting different features of DALI and how to use DALI to interface with deep learning frameworks.

Also note:

* Documentation for the latest stable release is available [here](https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading), and
* Nightly version of the documentation that stays in sync with the master branch is available [here](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/index.html).

----

## Additional resources


- GPU Technology Conference 2018 presentation about DALI, T. Gale, S. Layton and P. Tredak: [Slides](http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf), [Recording](http://on-demand.gputechconf.com/gtc/2018/video/S8906/).

----

## Contributing to DALI


We welcome contributions to DALI. To contribute to DALI and make pull requests, follow the guidelines outlined in the [Contributing](https://github.com/rajkaramchedu-nvidia/DALI/blob/master/CONTRIBUTING.md) document.

If you are looking for a task good for the start please check one from [external contribution welcome label](https://github.com/NVIDIA/DALI/labels/external%20contribution%20welcome).

## Reporting problems, asking questions


We appreciate feedback, questions or bug reports. When you need help with the code, follow the process outlined in the Stack Overflow (https://stackoverflow.com/help/mcve) document. Ensure that the posted examples are:

* **minimal**: Use as little code as possible that still produces the same problem.
* **complete**: Provide all parts needed to reproduce the problem. Check if you can strip external dependency and still show the problem. The less time we spend on reproducing the problems, the more time we can dedicate  to the fixes.
* **verifiable**: Test the code you are about to provide, to make sure that it reproduces the problem. Remove all other problems that are not related to your request.


## Contributors


DALI is being built with major contributions from Trevor Gale, Przemek Tredak, Simon Layton, Andrei Ivanov, Serge Panev.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Documentation](https://img.shields.io/badge/Nvidia%20DALI-documentation-brightgreen.svg?longCache=true)](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/)