Installation
============

Prerequisites
-------------

.. |driver link| replace:: **NVIDIA Driver**
.. _driver link: https://www.nvidia.com/drivers
.. |cuda link| replace:: **NVIDIA CUDA 10.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |mxnet link| replace:: **MXNet 1.3**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **PyTorch 0.4**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow 1.7**
.. _tf link: https://www.tensorflow.org
.. |compatibility link| replace:: enhanced CUDA compatibility guide
.. _compatibility link : https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases

1. Linux x64.
2. |driver link|_ supporting `CUDA 10.0 <https://developer.nvidia.com/cuda-downloads>`__ or later (i.e., 410.48 or later driver releases).
3. (Optional) One or more of the following deep learning frameworks:

  - |mxnet link|_ ``mxnet-cu100`` or later.
  - |pytorch link|_ or later.
  - |tf link|_ or later.


DALI in NGC Containers
----------------------

DALI is preinstalled in the TensorFlow, PyTorch, and MXNet containers in versions 18.07 and
later on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.

----

pip - Official Releases
-----------------------


nvidia-dali
^^^^^^^^^^^

Execute the following command to install the latest DALI for specified CUDA version (please check
:doc:`support matrix <support_matrix>` to see if your platform is supported):

* for CUDA 10.2:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102

* for CUDA 11.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

.. note::

  CUDA 11.0 build uses CUDA toolkit enhanced compatibility. It is built with the latest CUDA 11.x
  toolkit while it can run on the latest, stable CUDA 11.0 capable drivers (450.80 or later).
  Using the latest driver may enable additional functionality. More details can be found in
  |compatibility link|_.

.. note::

  Please always use the latest version of pip available (at least >= 19.3) and update when possible by issuing `pip install --upgrade pip`

nvidia-dali-tf-plugin
^^^^^^^^^^^^^^^^^^^^^

DALI doesn't contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package
which will be built against the currently installed version of TensorFlow:

* for CUDA 10.2:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda102

* for CUDA 11.0:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110


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

* for CUDA 10.2:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda102
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda102

* for CUDA 11.0:

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-tf-plugin-nightly-cuda110


Weekly Builds
^^^^^^^^^^^^^

Also, there is a weekly release channel with more thorough testing. To access most recent weekly
builds please use the following release channel (available only for CUDA 11):

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-weekly-cuda110
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly --upgrade nvidia-dali-tf-plugin-weekly-cuda110


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

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/11.0 --upgrade nvidia-dali
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/11.0 --upgrade nvidia-dali-tf-plugin

CUDA 9 build is provided up to DALI 0.22.0.

CUDA 10 build is provided up to DALI 1.3.0.

CUDA 10.2 build is provided starting from DALI 1.4.0.

CUDA 11 build is provided starting from DALI 0.22.0.

Open Cognitive Environment (Open-CE)
------------------------------------

.. |oce link| replace:: **external organizations**
.. _oce link: https://github.com/open-ce/open-ce#community-builds

DALI is also available as a part of the Open Cognitive Environment - a project that contains everything
that is needed to build conda packages for a collection of machine learning and deep learning frameworks.

This effort is community-driven and the DALI version available there may not be up to date.

Prebuild packages (including DALI) are hosted by |oce link|_.

