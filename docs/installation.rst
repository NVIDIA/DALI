Installation
============

DALI and NGC
------------

DALI is preinstalled in the `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_ TensorFlow, PyTorch, and MXNet containers in versions 18.07 and later.

----

Installing prebuilt DALI packages
---------------------------------

Prerequisites
^^^^^^^^^^^^^


.. |driver link| replace:: **NVIDIA Driver**
.. _driver link: https://www.nvidia.com/drivers
.. |cuda link| replace:: **NVIDIA CUDA 9.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |mxnet link| replace:: **MXNet 1.3**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **PyTorch 0.4**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow 1.7**
.. _tf link: https://www.tensorflow.org

1. Linux x64.
2. |driver link|_ supporting `CUDA 9.0 <https://developer.nvidia.com/cuda-downloads>`__ or later (i.e., 384.xx or later driver releases).
3. One or more of the following deep learning frameworks:

  - |mxnet link|_ ``mxnet-cu90`` or later.
  - |pytorch link|_ or later.
  - |tf link|_ or later.


Installation
^^^^^^^^^^^^
To install a CUDA 9.0 based DALI build, execute the following command:

.. code-block:: bash

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda90

For CUDA 10.0 based build use:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

.. note::

  Since version 0.6.1, DALI doesn't contain prebuilt versions of the DALI TensorFlow plugin. It needs to be installed as a separate package which will be built against the currently installed version of TensorFlow:

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda90

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda100

Installing this package will install ``nvidia-dali-cudaXXX`` and its dependencies, if these dependencies are not already installed. The package ``tensorflow-gpu`` must be installed before attempting to install ``nvidia-dali-tf-plugin-cudaXXX``.

.. note::

  The package ``nvidia-dali-tf-plugin-cudaXXX`` has a strict requirement with ``nvidia-dali-cudaXXX`` as its exact same version.
  Thus, installing ``nvidia-dali-tf-plugin-cudaXXX`` at its latest version will replace any older ``nvidia-dali-cudaXXX`` versions already installed with the latest.
  To work with older versions of DALI, provide the version explicitly to the ``pip install`` command.

.. code-block:: bash

    CUDA_SUFFIX=100
    CURRENT_DALI_VERSION=0.23  # Already installed version of nvidia-dali-cuda${CUDA_SUFFIX}, assuming we don't want to upgrade
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-tf-plugin-cuda${CUDA_SUFFIX}==$CURRENT_DALI_VERSION


For older versions of DALI (0.22 and lower), use the package `nvidia-dali`. The CUDA version is selected by changing the pip index:

.. code-block:: bash

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali-tf-plugin

.. code-block:: bash

   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali-tf-plugin


Pre-built packages in Watson Machine Learing Community Edition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. |wmlce link| replace:: **WML CE installation**
.. _wmlce link: https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.1/navigation/wmlce_install.html

IBM publishes pre-built DALI packages as part of Watson Machine Learning Community Edition (WML CE). WML CE includes conda packages for both IBM Power and x86 systems. The initial release includes DALI 0.9 built against CUDA 10.1 and with TensorFlow support. Other versions may be added in the future. The WML CE conda channel also includes the CUDA prerequisites for DALI.

After installing conda and configuring the WML CE conda channel (see |wmlce link|_) you can install DALI:

.. code-block:: bash

    $ conda create -y -n my-dali-env python=3.6 dali

    $ conda activate my-dali-env

    (my-dali-env) $ conda list dali
    ...
    dali                      0.9             py36_666ce55_1094.g70c071f

Nightly and weekly release channels
"""""""""""""""""""""""""""""""""""

.. note::

  While binaries available to download from nightly and weekly builds include most recent changes
  available in the GitHub some functionalities may not work or provide inferior performance comparing
  to the official releases. Those builds are meant for the early adopters seeking for the most recent
  version available and being ready to boldly go where no man has gone before.

.. note::

  It is recommended to uninstall regular DALI and TensorFlow plugin before installing nvidia-dali-nightly
  or nvidia-dali-weekly as they are installed in the same path

Nightly builds
**************

To access most recent nightly builds please use flowing release channel:

* for CUDA9

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly/cuda/9.0 nvidia-dali-nightly
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly/cuda/9.0 nvidia-dali-tf-plugin-nightly

* for CUDA10

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly/cuda/10.0 nvidia-dali-nightly
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly/cuda/10.0 nvidia-dali-tf-plugin-nightly

Weekly builds
**************

Also, there is a weekly release channel with more thorough testing (only CUDA10 builds are provided there):

.. code-block:: bash

  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly/cuda/10.0 nvidia-dali-weekly
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly/cuda/10.0 nvidia-dali-tf-plugin-weekly
