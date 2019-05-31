|License|  |Documentation|

NVIDIA DALI
===========
.. overview-begin-marker-do-not-remove

Deep learning applications require complex, multi-stage pre-processing
data pipelines. Such data pipelines involve compute-intensive operations
that are carried out on the CPU. For example, tasks such as: load data
from disk, decode, crop, random resize, color and spatial augmentations
and format conversions, are mainly carried out on the CPUs, limiting the
performance and scalability of training and inference.

In addition, the deep learning frameworks have multiple data
pre-processing implementations, resulting in challenges such as
portability of training and inference workflows, and code
maintainability.

NVIDIA Data Loading Library (DALI) is a collection of highly optimized
building blocks, and an execution engine, to accelerate the
pre-processing of the input data for deep learning applications. DALI
provides both the performance and the flexibility for accelerating
different data pipelines as a single library. This single library can
then be easily integrated into different deep learning training and
inference applications.

Highlights
----------

- Full data pipeline--accelerated from reading the disk to getting
  ready for training and inference.
- Flexibility through configurable graphs and custom operators.
- Support for image classification and segmentation workloads.
- Ease of integration through direct framework plugins and open
  source bindings.
- Portable training workflows with multiple input formats--JPEG,
  PNG (fallback to CPU), TIFF (fallback to CPU), BMP (fallback to CPU),
  raw formats, LMDB, RecordIO, TFRecord.
- Extensible for user-specific needs through open source license.

.. overview-end-marker-do-not-remove

----

Table of Contents
-----------------

- `Installation <https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/quickstart.html>`_
- `Getting started <https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/getting%20started.html>`_
- `Examples`_
- `Additional resources`_
- `Contributing to DALI`_
- `Reporting problems, asking questions`_
- `Contributors`_

----

Examples
--------

The `docs/examples <https://github.com/NVIDIA/DALI/blob/master/docs/examples>`_
directory contains a few examples (in the form of Jupyter notebooks)
highlighting different features of DALI and how to use DALI to interface
with deep learning frameworks.

Also note:

- Documentation for the latest stable release is available
  |here1|_, and
- Nightly version of the documentation that stays in sync with the
  master branch is available |here2|_.

.. |here1| replace:: here
.. _here1: https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading
.. |here2| replace:: here
.. _here2: https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/index.html

----

Additional resources
--------------------

- GPU Technology Conference 2018; Fast data pipeline for deep learning training, T. Gale, S. Layton and P. Trędak: |slides1|_, |recording1|_.
- GPU Technology Conference 2019; Fast AI data pre-preprocessing with DALI; Janusz Lisiecki, Michał Zientkiewicz: |slides2|_, |recording2|_.
- GPU Technology Conference 2019; Integration of DALI with TensorRT on Xavier; Josh Park and Anurag Dixit: |slides3|_, |recording3|_.
- `Developer page <https://developer.nvidia.com/DALI>`_.
- `Blog post <https://devblogs.nvidia.com/fast-ai-data-preprocessing-with-nvidia-dali/>`_.

.. |slides1| replace:: slides
.. _slides1:  http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf
.. |recording1| replace:: recording
.. _recording1: http://on-demand.gputechconf.com/gtc/2018/video/S8906/
.. |slides2| replace:: slides
.. _slides2:  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9925-fast-ai-data-pre-processing-with-nvidia-dali.pdf
.. |recording2| replace:: recording
.. _recording2: https://developer.nvidia.com/gtc/2019/video/S9925/video
.. |slides3| replace:: slides
.. _slides3:  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9818-integration-of-tensorrt-with-dali-on-xavier.pdf
.. |recording3| replace:: recording
.. _recording3: https://developer.nvidia.com/gtc/2019/video/S9818/video

----

Contributing to DALI
--------------------

We welcome contributions to DALI. To contribute to DALI and make pull requests,
follow the guidelines outlined in the `Contributing <https://github.com/NVIDIA/DALI/blob/master/CONTRIBUTING.md>`_
document.

If you are looking for a task good for the start please check one from
`external contribution welcome label <https://github.com/NVIDIA/DALI/labels/external%20contribution%20welcome>`_.

Reporting problems, asking questions
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

Contributors
------------

DALI is being built with major contributions from Trevor Gale, Przemek
Tredak, Simon Layton, Andrei Ivanov, Serge Panev.

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. |Documentation| image:: https://img.shields.io/badge/Nvidia%20DALI-documentation-brightgreen.svg?longCache=true
   :target: https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/ides
