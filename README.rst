|License|  |Documentation|

NVIDIA DALI
===========
.. overview-begin-marker-do-not-remove

The NVIDIA Data Loading Library (DALI) is a library for data loading and
pre-processing to accelerate deep learning applications. It provides a
collection of highly optimized building blocks for loading and processing
image, video and audio data. It can be used as a portable drop-in replacement
for built in data loaders and data iterators in popular deep learning frameworks.

Deep learning applications require complex, multi-stage data processing pipelines
that include loading, decoding, cropping, resizing, and many other augmentations.
These data processing pipelines, which are currently executed on the CPU, have become a
bottleneck, limiting the performance and scalability of training and inference.

DALI addresses the problem of the CPU bottleneck by offloading data preprocessing to the
GPU. Additionally, DALI relies on its own execution engine, built to maximize the throughput
of the input pipeline. Features such as prefetching, parallel execution, and batch processing
are handled transparently for the user.

In addition, the deep learning frameworks have multiple data pre-processing implementations,
resulting in challenges such as portability of training and inference workflows, and code
maintainability. Data processing pipelines implemented using DALI are portable because they
can easily be retargeted to TensorFlow, PyTorch, MXNet and PaddlePaddle.

.. image:: /dali.png
    :width: 800
    :align: center
    :alt: DALI Diagram

Highlights
----------
- Easy-to-use functional style Python API.
- Multiple data formats support - LMDB, RecordIO, TFRecord, COCO, JPEG, JPEG 2000, WAV, FLAC, OGG, H.264, VP9 and HEVC.
- Portable across popular deep learning frameworks: TensorFlow, PyTorch, MXNet, PaddlePaddle.
- Supports CPU and GPU execution.
- Scalable across multiple GPUs.
- Flexible graphs let developers create custom pipelines.
- Extensible for user-specific needs with custom operators.
- Accelerates image classification (ResNet-50), object detection (SSD) workloads as well as ASR models (Jasper, RNN-T).
- Allows direct data path between storage and GPU memory with |gds|_.
- Easy integration with |triton|_ with |triton-dali-backend|_.
- Open source.

.. |gds| replace:: GPUDirect Storage
.. _gds: https://developer.nvidia.com/gpudirect-storage

.. |triton| replace:: NVIDIA Triton Inference Server
.. _triton: https://developer.nvidia.com/nvidia-triton-inference-server

.. |triton-dali-backend| replace:: DALI TRITON Backend
.. _triton-dali-backend: https://github.com/triton-inference-server/dali_backend

.. overview-end-marker-do-not-remove

----

DALI Roadmap
------------

|dali-roadmap-link|_ a high-level overview of our 2021 plan. You should be aware that this
roadmap may change at any time and the order below does not reflect any type of priority.

We strongly encourage you to comment on our roadmap and provide us feedback on the mentioned
GitHub issue.

.. |dali-roadmap-link| replace:: The following issue represents
.. _dali-roadmap-link: https://github.com/NVIDIA/DALI/issues/2978

----

Installing DALI
---------------

To install the latest DALI release for the latest CUDA version (11.x)::

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

DALI comes preinstalled in the TensorFlow, PyTorch, and MXNet containers on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_
(versions 18.07 and later).

For other installation paths (TensorFlow plugin, older CUDA version, nightly and weekly builds, etc),
please refer to the |docs_install|_.

To build DALI from source, please refer to the |dali_compile|_.

.. |docs_install| replace:: Installation Guide
.. _docs_install: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
.. |dali_compile| replace:: Compilation Guide
.. _dali_compile: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/compilation.html

----

Examples and Tutorials
----------------------

An introduction to DALI can be found in the |dali_start|_ page.

More advanced examples can be found in the |dali_examples|_ page.

For an interactive version (Jupyter notebook) of the examples, go to the `docs/examples <https://github.com/NVIDIA/DALI/blob/main/docs/examples>`_
directory.

**Note:** Select the |release-doc|_ or the |nightly-doc|_, which stays in sync with the main branch,
depending on your version.

.. |dali_start| replace:: Getting Started
.. _dali_start: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting%20started.html
.. |dali_examples| replace:: Examples and Tutorials
.. _dali_examples: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/index.html
.. |release-doc| replace:: Latest Release Documentation
.. _release-doc: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html
.. |nightly-doc| replace:: Nightly Release Documentation
.. _nightly-doc: https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/index.html

----

Additional Resources
--------------------

- GPU Technology Conference 2021; **NVIDIA DALI: GPU-Powered Data Preprocessing** by Krzysztof Łęcki and Michał Szołucha: |event2021|_.
- GPU Technology Conference 2020; **Fast Data Pre-Processing with NVIDIA Data Loading Library (DALI)**; Albert Wolant, Joaquin Anton Guirao |recording4|_.
- GPU Technology Conference 2019; **Fast AI data pre-preprocessing with DALI**; Janusz Lisiecki, Michał Zientkiewicz: |slides2|_, |recording2|_.
- GPU Technology Conference 2019; **Integration of DALI with TensorRT on Xavier**; Josh Park and Anurag Dixit: |slides3|_, |recording3|_.
- GPU Technology Conference 2018; **Fast data pipeline for deep learning training**, T. Gale, S. Layton and P. Trędak: |slides1|_, |recording1|_.
- `Developer Page <https://developer.nvidia.com/DALI>`_.
- `Blog Posts <https://developer.nvidia.com/blog/tag/dali/>`_.

.. |slides1| replace:: slides
.. _slides1:  http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf
.. |recording1| replace:: recording
.. _recording1: https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8906/
.. |slides2| replace:: slides
.. _slides2:  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9925-fast-ai-data-pre-processing-with-nvidia-dali.pdf
.. |recording2| replace:: recording
.. _recording2: https://developer.nvidia.com/gtc/2019/video/S9925/video
.. |slides3| replace:: slides
.. _slides3:  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9818-integration-of-tensorrt-with-dali-on-xavier.pdf
.. |recording3| replace:: recording
.. _recording3: https://developer.nvidia.com/gtc/2019/video/S9818/video
.. |recording4| replace:: recording
.. _recording4: https://developer.nvidia.com/gtc/2020/video/s21139
.. |event2021| replace:: event
.. _event2021:  https://gtc21.event.nvidia.com/media/1_j4dk7w7q

----

Contributing to DALI
--------------------

We welcome contributions to DALI. To contribute to DALI and make pull requests,
follow the guidelines outlined in the `Contributing <https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md>`_
document.

If you are looking for a task good for the start please check one from
`external contribution welcome label <https://github.com/NVIDIA/DALI/labels/external%20contribution%20welcome>`_.

Reporting Problems, Asking Questions
------------------------------------

We appreciate feedback, questions or bug reports. When you need help
with the code, follow the process outlined in the Stack Overflow
`<https://stackoverflow.com/help/mcve>`_ document. Ensure that the
posted examples are:

- **minimal**: Use as little code as possible that still produces the same problem.
- **complete**: Provide all parts needed to reproduce the problem.
  Check if you can strip external dependency and still show the problem.
  The less time we spend on reproducing the problems, the more time we
  can dedicate to the fixes.
- **verifiable**: Test the code you are about to provide, to make sure
  that it reproduces the problem. Remove all other problems that are not
  related to your request.

Acknowledgements
----------------

DALI was originally built with major contributions from Trevor Gale, Przemek Tredak,
Simon Layton, Andrei Ivanov and Serge Panev.

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. |Documentation| image:: https://img.shields.io/badge/Nvidia%20DALI-documentation-brightgreen.svg?longCache=true
   :target: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html
