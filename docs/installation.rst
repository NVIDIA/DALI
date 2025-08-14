Installation
============

Prerequisites
-------------

1. Linux x64.
2. `**NVIDIA Driver** <https://www.nvidia.com/drivers>`_ supporting `CUDA 12.0 <https://developer.nvidia.com/cuda-downloads>`__
   or later (i.e. 525.60 or later driver releases).
3. `**CUDA Toolkit** <https://developer.nvidia.com/cuda-downloads>`_ - the toolkit is linked
   dynamically and it is required to be installed.
4. [Optional] One or more of the following deep learning frameworks:

   * `PyTorch <https://pytorch.org>`__
   * `TensorFlow <https://www.tensorflow.org>`__
   * `JAX <https://github.com/google/jax>`__
   * `PaddlePaddle <https://www.paddlepaddle.org.cn/en>`__


DALI in NGC Containers
----------------------

DALI is preinstalled in the `TensorFlow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_,
`PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_,
and `PaddlePaddle <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paddlepaddle>`_
containers on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.

----

.. _pip wheels:

pip - Official Releases
-----------------------


nvidia-dali
^^^^^^^^^^^

Execute the following command to install the latest DALI for specified CUDA version (please check
:doc:`support matrix <support_matrix>` to see if your platform is supported):

* for CUDA 12.0:

.. code-block:: bash

   pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120

or just

.. code-block:: bash

   pip install nvidia-dali-cuda120

* for CUDA 13.0:

.. code-block:: bash

   pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda130

or just

.. code-block:: bash

   pip install nvidia-dali-cuda130

.. note::

  CUDA 12.0 and CUDA 13.0 build uses CUDA toolkit enhanced compatibility. It is built with the latest CUDA 12.x/13.x respectively
  toolkit while it can run on the latest, stable CUDA 12.0 and CUDA 13.0 capable drivers (525.60 or later and 580.x or later respectively).
  Using the latest driver may enable additional functionality. More details can be found in
  `enhanced CUDA compatibility guide <https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases>`_.

.. note::

  Please always use the latest version of pip available (at least >= 19.3) and update when possible by issuing `pip install --upgrade pip`

nvidia-dali-tf-plugin
^^^^^^^^^^^^^^^^^^^^^

DALI doesn't contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package
which will be built against the currently installed version of TensorFlow:

* for CUDA 12.0:

.. code-block:: bash

   pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-tf-plugin-cuda120

or just

.. code-block:: bash

   pip install nvidia-dali-tf-plugin-cuda120

* for CUDA 13.0:

.. code-block:: bash

   pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-tf-plugin-cuda130

or just

.. code-block:: bash

   pip install nvidia-dali-tf-plugin-cuda130

Installing this package will install ``nvidia-dali-cudaXXX`` and its dependencies, if they are not already installed. The package ``tensorflow`` must be installed before attempting to install ``nvidia-dali-tf-plugin-cudaXXX``.

.. note::

  The packages ``nvidia-dali-tf-plugin-cudaXXX`` and ``nvidia-dali-cudaXXX`` should be in exactly the same version.
  Therefore, installing the latest ``nvidia-dali-tf-plugin-cudaXXX``, will replace any older ``nvidia-dali-cudaXXX`` version already installed.
  To work with older versions of DALI, provide the version explicitly to the ``pip install`` command.

pip - Nightly and Weekly Releases
---------------------------------

.. note::

  While binaries available to download from nightly and weekly builds include most recent changes
  available in the GitHub some functionalities may not work or provide inferior performance comparing
  to the official releases. Those builds are meant for the early adopters seeking for the most recent
  version available and being ready to boldly go where no man has gone before.

.. note::

  It is recommended to uninstall regular DALI and TensorFlow plugin before installing nightly or weekly
  builds as they are installed in the same path

Nightly Builds
^^^^^^^^^^^^^^

To access most recent nightly builds please use flowing release channel:

* for CUDA 12.0:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda120
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda120

* for CUDA 13.0:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda130
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda130


Weekly Builds
^^^^^^^^^^^^^

Also, there is a weekly release channel with more thorough testing. To access most recent weekly
builds please use the following release channel (available only for CUDA 13):

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-weekly-cuda130
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-tf-plugin-weekly-cuda130


pip - Legacy Releases
---------------------

For older versions of DALI (0.22 and lower), use the package `nvidia-dali`. The CUDA version can be selected by changing the pip index:

.. code-block:: bash

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 --upgrade nvidia-dali
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 --upgrade nvidia-dali-tf-plugin

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 --upgrade nvidia-dali
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 --upgrade nvidia-dali-tf-plugin

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda102

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/11.0 --upgrade nvidia-dali
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/11.0 --upgrade nvidia-dali-tf-plugin

   pip install --upgrade nvidia-dali-cuda110
   pip install --upgrade nvidia-dali-tf-plugin-cuda110

CUDA 12 build is provided starting from DALI 1.22.0.

CUDA 11 build is provided starting from DALI 0.22.0.

CUDA 10.2 build is provided starting from DALI 1.4.0 up to DALI 1.20.

CUDA 10 build is provided up to DALI 1.3.0.

CUDA 9 build is provided up to DALI 0.22.0.

Open Cognitive Environment (Open-CE)
------------------------------------

DALI is also available as a part of the Open Cognitive Environment - a project that contains everything
that is needed to build conda packages for a collection of machine learning and deep learning frameworks.

This effort is community-driven and the DALI version available there may not be up to date.

Prebuild packages (including DALI) are hosted by `**external organizations** <https://github.com/open-ce/open-ce#community-builds>`_.

Conda conda-forge
-----------------

DALI is available as part of the conda-forge ecosystem.

This effort is community-driven and the DALI version available there may not be up to date.

`The package is available here <https://anaconda.org/conda-forge/nvidia-dali-python>`_.