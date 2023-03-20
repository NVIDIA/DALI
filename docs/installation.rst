Installation
============

Prerequisites
-------------

.. |driver link| replace:: **NVIDIA Driver**
.. _driver link: https://www.nvidia.com/drivers
.. |cuda link| replace:: **NVIDIA CUDA 11.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |cuda toolkit link| replace:: **CUDA Toolkit**
.. _cuda toolkit link: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
.. |cuda link| replace:: **NVIDIA CUDA 11.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |mxnet link| replace:: **MXNet**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **PyTorch**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow**
.. _tf link: https://www.tensorflow.org
.. |pddl link| replace:: **PaddlePaddle**
.. _pddl link: https://www.paddlepaddle.org.cn
.. |compatibility link| replace:: enhanced CUDA compatibility guide
.. _compatibility link : https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases

1. Linux x64.
2. |driver link|_ supporting `CUDA 11.0 <https://developer.nvidia.com/cuda-downloads>`__ or later (i.e. 450.80.02 or later driver releases).
3. |cuda toolkit link|_ - for DALI based on CUDA 12, the toolkit is linked dynamically and it is required to be installed. For CUDA 11 builds it is optional.
4. [Optional] One or more of the following deep learning frameworks:

  - |mxnet link|_
  - |pytorch link|_
  - |tf link|_
  - |pddl link|_


DALI in NGC Containers
----------------------

DALI is preinstalled in the `TensorFlow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_,
`PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_,
`NVIDIA Optimized Deep Learning Framework, powered by Apache MXNet <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/mxnet>`_,
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

* for CUDA 11.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

* for CUDA 12.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

.. note::

  CUDA 11.0 and CUDA 12.0 build uses CUDA toolkit enhanced compatibility. It is built with the latest CUDA 11.x/12.x respectively
  toolkit while it can run on the latest, stable CUDA 11.0 and CUDA 12.0 capable drivers (450.80 or later and 525.60 or later respectively).
  Using the latest driver may enable additional functionality. More details can be found in
  |compatibility link|_.

.. note::

  Please always use the latest version of pip available (at least >= 19.3) and update when possible by issuing `pip install --upgrade pip`

nvidia-dali-tf-plugin
^^^^^^^^^^^^^^^^^^^^^

DALI doesn't contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package
which will be built against the currently installed version of TensorFlow:

* for CUDA 11.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110

* for CUDA 12.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda120


Installing this package will install ``nvidia-dali-cudaXXX`` and its dependencies, if they are not already installed. The package ``tensorflow-gpu`` must be installed before attempting to install ``nvidia-dali-tf-plugin-cudaXXX``.

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

* for CUDA 11.0:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda110

* for CUDA 12.0:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda120
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda120


Weekly Builds
^^^^^^^^^^^^^

Also, there is a weekly release channel with more thorough testing. To access most recent weekly
builds please use the following release channel (available only for CUDA 12):

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-weekly-cuda120
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-tf-plugin-weekly-cuda120


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

CUDA 11 build is provided starting from DALI 0.22.0.

CUDA 10.2 build is provided starting from DALI 1.4.0 up to DALI 1.20.

CUDA 10 build is provided up to DALI 1.3.0.

CUDA 9 build is provided up to DALI 0.22.0.

Open Cognitive Environment (Open-CE)
------------------------------------

.. |oce link| replace:: **external organizations**
.. _oce link: https://github.com/open-ce/open-ce#community-builds

DALI is also available as a part of the Open Cognitive Environment - a project that contains everything
that is needed to build conda packages for a collection of machine learning and deep learning frameworks.

This effort is community-driven and the DALI version available there may not be up to date.

Prebuild packages (including DALI) are hosted by |oce link|_.

