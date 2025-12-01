|License|  |Documentation|  |Format|

NVIDIA DALI
===========
.. overview-begin-marker-do-not-remove

The NVIDIA Data Loading Library (DALI) is a GPU-accelerated library for data loading
and pre-processing to accelerate deep learning applications. It provides a
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
can easily be retargeted to TensorFlow, PyTorch, and PaddlePaddle.

.. image:: /dali.png
    :width: 800
    :align: center
    :alt: DALI Diagram

DALI in action:

.. code-block:: python

  from nvidia.dali.pipeline import pipeline_def
  import nvidia.dali.types as types
  import nvidia.dali.fn as fn
  from nvidia.dali.plugin.pytorch import DALIGenericIterator
  import os

  # To run with different data, see documentation of nvidia.dali.fn.readers.file
  # points to https://github.com/NVIDIA/DALI_extra
  data_root_dir = os.environ['DALI_EXTRA_PATH']
  images_dir = os.path.join(data_root_dir, 'db', 'single', 'jpeg')


  def loss_func(pred, y):
      pass


  def model(x):
      pass


  def backward(loss, model):
      pass


  @pipeline_def(num_threads=4, device_id=0)
  def get_dali_pipeline():
      images, labels = fn.readers.file(
          file_root=images_dir, random_shuffle=True, name="Reader")
      # decode data on the GPU
      images = fn.decoders.image_random_crop(
          images, device="mixed", output_type=types.RGB)
      # the rest of processing happens on the GPU as well
      images = fn.resize(images, resize_x=256, resize_y=256)
      images = fn.crop_mirror_normalize(
          images,
          crop_h=224,
          crop_w=224,
          mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
          std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
          mirror=fn.random.coin_flip())
      return images, labels


  train_data = DALIGenericIterator(
      [get_dali_pipeline(batch_size=16)],
      ['data', 'label'],
      reader_name='Reader'
  )


  for i, data in enumerate(train_data):
      x, y = data[0]['data'], data[0]['label']
      pred = model(x)
      loss = loss_func(pred, y)
      backward(loss, model)


Highlights
----------
- Easy-to-use functional style Python API.
- Multiple data formats support - LMDB, RecordIO, TFRecord, COCO, JPEG, JPEG 2000, WAV, FLAC, OGG, H.264, VP9 and HEVC.
- Portable across popular deep learning frameworks: TensorFlow, PyTorch, PaddlePaddle, JAX.
- Supports CPU and GPU execution.
- Scalable across multiple GPUs.
- Flexible graphs let developers create custom pipelines.
- Extensible for user-specific needs with custom operators.
- Accelerates image classification (ResNet-50), object detection (SSD) workloads as well as ASR models (Jasper, RNN-T).
- Allows direct data path between storage and GPU memory with `GPUDirect Storage <https://developer.nvidia.com/gpudirect-storage>`__.
- Easy integration with `NVIDIA Triton Inference Server <https://developer.nvidia.com/nvidia-triton-inference-server>`__
  with `DALI TRITON Backend <https://github.com/triton-inference-server/dali_backend>`__.
- Open source.

.. overview-end-marker-do-not-remove

----

DALI success stories:
---------------------

- `During Kaggle computer vision competitions <https://www.kaggle.com/code/theoviel/rsna-breast-baseline-faster-inference-with-dali>`__:
  `"DALI is one of the best things I have learned in this competition" <https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391059>`__
- `Lightning Pose - state of the art pose estimation research model <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168383/>`__
- `To improve the resource utilization in Advanced Computing Infrastructure <https://arcwiki.rs.gsu.edu/en/dali/using_nvidia_dali_loader>`__
- `MLPerf - the industry standard for benchmarking compute and deep learning hardware and software <https://developer.nvidia.com/blog/mlperf-hpc-v1-0-deep-dive-into-optimizations-leading-to-record-setting-nvidia-performance/>`__
- `"we optimized major models inside eBay with the DALI framework" <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62578/>`__

----

DALI Roadmap
------------

`The following issue represents <https://github.com/NVIDIA/DALI/issues/5320>`__ a high-level overview of our 2024 plan. You should be aware that this
roadmap may change at any time and the order of its items does not reflect any type of priority.

We strongly encourage you to comment on our roadmap and provide us feedback on the mentioned
GitHub issue.

----

Installing DALI
---------------

To install the latest DALI release for the latest CUDA version (12.x)::

    pip install nvidia-dali-cuda120
    # or
    pip install --extra-index-url https://pypi.nvidia.com  --upgrade nvidia-dali-cuda120

DALI requires `NVIDIA driver <https://www.nvidia.com/drivers>`__ supporting the appropriate CUDA version.
In case of DALI based on CUDA 12, it requires `CUDA Toolkit <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__
to be installed.

DALI comes preinstalled in the `TensorFlow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`__,
`PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__,
and `PaddlePaddle <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paddlepaddle>`__
containers on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`__.

For other installation paths (TensorFlow plugin, older CUDA version, nightly and weekly builds, etc),
and specific requirements please refer to the `Installation Guide <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html>`__.

To build DALI from source, please refer to the `Compilation Guide <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/compilation.html>`__.


----

Examples and Tutorials
----------------------

An introduction to DALI can be found in the `Getting Started <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html>`__ page.

More advanced examples can be found in the `Examples and Tutorials <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/index.html>`__ page.

For an interactive version (Jupyter notebook) of the examples, go to the `docs/examples <https://github.com/NVIDIA/DALI/blob/main/docs/examples>`__
directory.

**Note:** Select the `Latest Release Documentation <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>`__
or the `Nightly Release Documentation <https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/index.html>`__, which stays in sync with the main branch,
depending on your version.

----

Additional Resources
--------------------

- GPU Technology Conference 2024; **Optimizing Inference Model Serving for Highest Performance at eBay**; Yiheng Wang:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62578/>`__
- GPU Technology Conference 2023; **Developer Breakout: Accelerating Enterprise Workflows With Triton Server and DALI**; Brandon Tuttle:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-se52140/>`__.
- GPU Technology Conference 2023; **GPU-Accelerating End-to-End Geospatial Workflows**; Kevin Green:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51796/>`__.
- GPU Technology Conference 2022; **Effective NVIDIA DALI: Accelerating Real-life Deep-learning Applications**; Rafał Banaś:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41442/>`__.
- GPU Technology Conference 2022; **Introduction to NVIDIA DALI: GPU-accelerated Data Preprocessing**; Joaquin Anton Guirao:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41443/>`__.
- GPU Technology Conference 2021; **NVIDIA DALI: GPU-Powered Data Preprocessing** by Krzysztof Łęcki and Michał Szołucha:
  `event <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31298/>`__.
- GPU Technology Conference 2020; **Fast Data Pre-Processing with NVIDIA Data Loading Library (DALI)**; Albert Wolant, Joaquin Anton Guirao:
  `recording <https://developer.nvidia.com/gtc/2020/video/s21139>`__.
- GPU Technology Conference 2019; **Fast AI data pre-preprocessing with DALI**; Janusz Lisiecki, Michał Zientkiewicz:
  `slides <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9925-fast-ai-data-pre-processing-with-nvidia-dali.pdf>`__,
  `recording <https://developer.nvidia.com/gtc/2019/video/S9925/video>`__.
- GPU Technology Conference 2019; **Integration of DALI with TensorRT on Xavier**; Josh Park and Anurag Dixit:
  `slides <https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9818-integration-of-tensorrt-with-dali-on-xavier.pdf>`__,
  `recording <https://developer.nvidia.com/gtc/2019/video/S9818/video>`__.
- GPU Technology Conference 2018; **Fast data pipeline for deep learning training**, T. Gale, S. Layton and P. Trędak:
  `slides <http://on-demand.gputechconf.com/gtc/2018/presentation/s8906-fast-data-pipelines-for-deep-learning-training.pdf>`__,
  `recording <https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8906/>`__.
- `Developer Page <https://developer.nvidia.com/DALI>`__.
- `Blog Posts <https://developer.nvidia.com/blog/tag/dali/>`__.


----

Contributing to DALI
--------------------

We welcome contributions to DALI. To contribute to DALI and make pull requests,
follow the guidelines outlined in the `Contributing <https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md>`__
document.

If you are looking for a task good for the start please check one from
`external contribution welcome label <https://github.com/NVIDIA/DALI/labels/external%20contribution%20welcome>`__.

Reporting Problems, Asking Questions
------------------------------------

We appreciate feedback, questions or bug reports. When you need help
with the code, follow the process outlined in the `Stack Overflow
<https://stackoverflow.com/help/mcve>`__ document. Ensure that the
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

.. |Documentation| image:: https://img.shields.io/badge/NVIDIA%20DALI-documentation-brightgreen.svg?longCache=true
   :target: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html

.. |Format| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
