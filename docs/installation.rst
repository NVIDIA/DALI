Installation
============

Select your configuration to get the install command. The sections below explain each option.

.. raw:: html
   :file: install_selector.html

Prerequisites
-------------

1. Linux x86_64 or aarch64 SBSA.
2. `NVIDIA driver <https://www.nvidia.com/drivers>`_ supporting
   `CUDA 12.0 <https://developer.nvidia.com/cuda-12-0-0-download-archive>`__ (525.60 or later) or
   `CUDA 13.0 <https://developer.nvidia.com/cuda-13-0-0-download-archive>`__ (580 or later).
3. For pip packages, the `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_, which DALI
   links dynamically.
4. For Docker, the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_,
   needed by ``docker run --gpus``.
5. [Optional] One or more of the following deep learning frameworks:

   * `PyTorch <https://pytorch.org>`__
   * `TensorFlow <https://www.tensorflow.org>`__
   * `JAX <https://jax.dev>`__
   * `PaddlePaddle <https://www.paddlepaddle.org.cn/en>`__

.. _pip wheels:
.. _install-pip:

pip
---

DALI wheels are built per CUDA major version: ``nvidia-dali-cuda120`` for CUDA 12 and
``nvidia-dali-cuda130`` for CUDA 13. They are published on PyPI and on https://pypi.nvidia.com,
for Linux x86_64 and aarch64 SBSA. Please check the :doc:`support matrix <support_matrix>` to see
if your platform is supported.

.. note::

  The CUDA 12 and CUDA 13 builds use CUDA enhanced compatibility. They are built with the latest CUDA 12.x
  and 13.x toolkits respectively, but run on any driver that supports CUDA 12.0 (525.60 or later) or
  CUDA 13.0 (580 or later). Using the latest driver may enable additional functionality. More details can
  be found in the `enhanced CUDA compatibility guide <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases>`_.

.. _install-tf-plugin:

TensorFlow Plugin
^^^^^^^^^^^^^^^^^

DALI doesn't contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package,
``nvidia-dali-tf-plugin-cudaXXX``, which will be built against the currently installed version of TensorFlow.
The package ``tensorflow`` must be installed before attempting to install ``nvidia-dali-tf-plugin-cudaXXX``.
Installing the plugin also installs ``nvidia-dali-cudaXXX`` and its dependencies, if they are not already installed.

.. note::

  Please always use the ``--no-build-isolation`` flag when installing the DALI TensorFlow plugin. The installation
  relies on TensorFlow being present in the environment, and pip enables build isolation by default, so this flag is
  necessary to install the right variant of the plugin for the installed TensorFlow version.

.. warning::

  The packages ``nvidia-dali-tf-plugin-cudaXXX`` and ``nvidia-dali-cudaXXX`` must have exactly the same version.
  Therefore, installing the latest ``nvidia-dali-tf-plugin-cudaXXX`` will replace any older ``nvidia-dali-cudaXXX`` version already installed.
  To work with older versions of DALI, provide the version explicitly to the ``pip install`` command.

.. _install-prerelease:

Nightly and Weekly Builds
^^^^^^^^^^^^^^^^^^^^^^^^^

Nightly builds are available for CUDA 12 and CUDA 13. There is also a weekly release channel with
more thorough testing, available only for CUDA 13.

.. note::

  Nightly and weekly builds include the most recent changes from GitHub, so some functionality may not
  work or may perform worse compared to the official releases. These builds are meant for early adopters
  seeking the most recent version and ready to boldly go where no one has gone before.

.. tip::

  It is recommended to uninstall the stable DALI and TensorFlow plugin packages before installing nightly or
  weekly builds, as they are installed in the same path.

.. _install-docker:

Docker
------

DALI is preinstalled in the `PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__
and `JAX <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax>`__ containers on
`NVIDIA GPU Cloud <https://ngc.nvidia.com>`__.

The `TensorFlow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`__ and
`PaddlePaddle <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paddlepaddle>`__ containers
are no longer released. Their final tags, ``25.02-tf2-py3`` and ``24.12-py3``, ship DALI 1.46.0
and 1.44.0 respectively. Install DALI with pip to get a newer version.

.. _install-conda:

Conda
-----

DALI is available on conda-forge as the `nvidia-dali-python <https://anaconda.org/conda-forge/nvidia-dali-python>`_ package.

DALI is also available as part of the Open Cognitive Environment (Open-CE), a project that contains everything
that is needed to build conda packages for a collection of machine learning and deep learning frameworks.
Prebuilt packages (including DALI) are hosted by `external organizations <https://github.com/open-ce/open-ce#community-builds>`_.

Both efforts are community-driven, and the DALI versions they provide may not be up to date.

.. _install-source:

From Source
-----------

To build DALI yourself, follow :doc:`compilation`. This is the only way to get DALI on Jetson
(see :ref:`jetson build`).

Older Versions
--------------

Each DALI release is built separately for each supported CUDA version. To install a specific version, pick the package for
your CUDA version from the tables below and pin the version, for example:

.. code-block:: bash

   pip install \
       --extra-index-url https://developer.download.nvidia.com/compute/redist \
       nvidia-dali-cuda110==1.50.0

A CUDA version written as ``13.x`` means the package runs on any CUDA 13 minor version.

Packages on PyPI:

.. list-table::
   :header-rows: 1

   * - CUDA
     - DALI versions
     - Package
   * - 13.x
     - 1.51.2 and later
     - ``nvidia-dali-cuda130``
   * - 12.x
     - 1.22.0 and later
     - ``nvidia-dali-cuda120``

Packages on ``https://developer.download.nvidia.com/compute/redist``:

.. list-table::
   :header-rows: 1

   * - CUDA
     - DALI versions
     - Package
   * - 11.x
     - 0.23.0 to 1.50.0 (1.16.0 and later also on PyPI)
     - ``nvidia-dali-cuda110``
   * - 10.2
     - 1.4.0 to 1.21.0
     - ``nvidia-dali-cuda102``
   * - 10.0
     - 0.23.0 to 1.3.0
     - ``nvidia-dali-cuda100``

For DALI 0.22.0 and earlier, the package is ``nvidia-dali``, with one index per CUDA version:

.. list-table::
   :header-rows: 1

   * - CUDA
     - DALI versions
     - Index
   * - 11.0
     - 0.22.0
     - ``https://developer.download.nvidia.com/compute/redist/cuda/11.0``
   * - 10.0
     - 0.8.0 to 0.22.0
     - ``https://developer.download.nvidia.com/compute/redist/cuda/10.0``
   * - 9.0
     - 0.1.0 to 0.22.0
     - ``https://developer.download.nvidia.com/compute/redist/cuda/9.0``

The TensorFlow plugin follows the same pattern (``nvidia-dali-tf-plugin-cudaXXX``, or
``nvidia-dali-tf-plugin`` for DALI 0.22.0 and earlier), in the same version as DALI.
