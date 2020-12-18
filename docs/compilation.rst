Compiling DALI from source
==========================

.. _DockerBuilderAnchor:

Compiling DALI from source (using Docker builder) - recommended
---------------------------------------------------------------

Following these steps, it is possible to recreate Python wheels in a similar fashion as we provide as an official prebuild binary.

Prerequisites
^^^^^^^^^^^^^

.. |docker link| replace:: **Docker**
.. _docker link: https://docs.docker.com/install/
.. |nvidia_docker| replace:: **NVIDIA Container Toolkit**
.. _nvidia_docker: https://github.com/NVIDIA/nvidia-docker

.. table::
   :align: center

   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | Linux x64                              |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |docker link|_                         | Follow installation guide and manual at the link (version 17.05 or later is required).      |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |nvidia_docker|_                       | Follow installation guide and manual at the link.                                           |
   |                                        |                                                                                             |
   |                                        | Using NVIDIA Container Toolkit is recommended as nvidia-docker2 is deprecated               |
   |                                        | but both are supported.                                                                     |
   |                                        |                                                                                             |
   |                                        | Required for building DALI TensorFlow Plugin.                                               |
   +----------------------------------------+---------------------------------------------------------------------------------------------+

Building Python wheel and (optionally) Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Change directory (``cd``) into ``docker`` directory and run ``./build.sh``. If needed,
set the following environment variables:

* | CUDA_VERSION - CUDA toolkit version (10.0, 11.0, 11.1 and 11.2).
  | The default is ``11.2``. Thanks to CUDA extended compatibility mode, CUDA 11.1 and 11.2 wheels are
    named as CUDA 11.0 because it can work with the CUDA 11.0 R450.x driver family. Please update
    to the latest recommended driver version in that family.
  | If the value of the CUDA_VERSION is prefixed with `.` then any value ``.XX.Y`` can be passed,
    the supported version check is suppressed, and the user needs to make sure that
    Dockerfile.cudaXXY.deps is present in the `docker/` directory.
* | NVIDIA_BUILD_ID - Custom ID of the build.
  | The default is ``1234``.
* | CREATE_WHL - Create a standalone wheel.
  | The default is ``YES``.
* | BUILD_TF_PLUGIN - Create a DALI TensorFlow plugin wheel as well.
  | The default is ``NO``.
* | PREBUILD_TF_PLUGINS - Whether to prebuild DALI TensorFlow plugin.
  | It should be used together
    with BUILD_TF_PLUGIN option. If both options are set to ``YES`` then DALI TensorFlow plugin
    package is built with prebuilt plugin binaries inside. If PREBUILD_TF_PLUGINS is set to
    ``NO`` then the wheel is still built but without prebuilding binaries - no prebuilt binaries
    are placed inside and the user needs to make sure that he has proper compiler version present
    (aligned with the one used to build present TensorFlow) so the plugin can be built during the
    installation of DALI TensorFlow plugin package. If is BUILD_TF_PLUGIN is set to ``NO``
    PREBUILD_TF_PLUGINS value is disregarded. The default is ``YES``.
* | CREATE_RUNNER - Create Docker image with cuDNN, CUDA and DALI installed inside.
  | It will create the ``Docker_run_cuda`` image, which needs to be run using |nvidia_docker|_
    and place the DALI wheel (and optionally the TensorFlow plugin if compiled) in the ``/opt/dali``
    directory.
  | The default is ``NO``.
* | PYVER - Python version used to create the runner image with DALI installed inside mentioned above.
  | The default is ``3.6``.
* DALI_BUILD_FLAVOR - adds a suffix to DALI package name and put a note about it in the whl package
  description, i.e. `nightly` will result in the `nvidia-dali-nightly`
* | CMAKE_BUILD_TYPE - build type, available options: Debug, DevDebug, Release, RelWithDebInfo.
  | The default is ``Release``.
* | STRIP_BINARY - when used with CMAKE_BUILD_TYPE equal to Debug, DevDebug, or RelWithDebInfo it
    produces bare wheel binary without any debug information and the second one with \*_debug.whl
    name with this information included.
  | In the case of the other build configurations, these two wheels will be identical.
* | BUILD_INHOST - ask docker to mount source code instead of copying it.
  | Thank to that consecutive builds are resuing existing object files and are faster
    for the development. Uses $DALI_BUILD_DIR as a directory for build objects. The default is ``YES``.
* | REBUILD_BUILDERS - if builder docker images need to be rebuild or can be reused from
    the previous build.
  | The default is ``NO``.
* | DALI_BUILD_DIR - where DALI build should happen.
  | It matters only bit the in-tree build where user may provide different path for every
    python/CUDA version. The default is ``build-docker-${CMAKE_BUILD_TYPE}-${PYV}-${CUDA_VERSION}``.
* | ARCH - architecture that DALI is build for, x86_64 and aarch64
    (SBSA - Server Base System Architecture) are supported.
  | The default is ``x86_64``.
* | WHL_PLATFORM_NAME - the name of the Python wheel platform tag.
  | The default is ``manylinux2014_x86_64``.

It is worth to mention that build.sh should accept the same set of environment variables as the project CMake.

The recommended command line is:

.. code-block:: bash

  CUDA_VERSION=Z ./build.sh

For example:

.. code-block:: bash

  CUDA_VERSION=11.1 ./build.sh

Will build CUDA 11.1 based DALI for Python 3 and place relevant Python wheel inside DALI_root/wheelhouse
The produced DALI wheel and TensorFlow Plugin are compatible with all Python versions supported by DALI.

----

Compiling DALI from source (bare metal)
---------------------------------------

Prerequisites
^^^^^^^^^^^^^

.. |cuda link| replace:: **NVIDIA CUDA 10.0**
.. _cuda link: https://developer.nvidia.com/cuda-downloads
.. |nvjpeg link| replace:: **nvJPEG library**
.. _nvjpeg link: https://developer.nvidia.com/nvjpeg
.. |protobuf link| replace:: **protobuf**
.. _protobuf link: https://github.com/google/protobuf
.. |cmake link| replace:: **CMake 3.13**
.. _cmake link: https://cmake.org
.. |jpegturbo link| replace:: **libjpeg-turbo 2.0.4 (2.0.3 for conda due to availability)**
.. _jpegturbo link: https://github.com/libjpeg-turbo/libjpeg-turbo
.. |libtiff link| replace:: **libtiff 4.1.0**
.. _libtiff link: http://libtiff.org/
.. |ffmpeg link| replace:: **FFmpeg 4.2.2**
.. _ffmpeg link: https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-4.2.2.tar.bz2
.. |libsnd link| replace:: **libsnd 1.0.28**
.. _libsnd link: https://developer.download.nvidia.com/compute/redist/nvidia-dali/libsndfile-1.0.28.tar.gz
.. |opencv link| replace:: **OpenCV 4**
.. _opencv link: https://opencv.org
.. |lmdb link| replace:: **liblmdb 0.9.x**
.. _lmdb link: https://github.com/LMDB/lmdb
.. |gcc link| replace:: **GCC 5.3.1**
.. _gcc link: https://www.gnu.org/software/gcc/
.. |boost link| replace:: **Boost 1.66**
.. _boost link: https://www.boost.org/

.. |mxnet link| replace:: **MXNet 1.5**
.. _mxnet link: http://mxnet.incubator.apache.org
.. |pytorch link| replace:: **PyTorch 1.1**
.. _pytorch link: https://pytorch.org
.. |tf link| replace:: **TensorFlow 1.12**
.. _tf link: https://www.tensorflow.org
.. |clang link| replace:: **clang**
.. _clang link: https://apt.llvm.org/
.. |gds link| replace:: **GPU Direct Storage**
.. _gds link: https://developer.nvidia.com/gpudirect-storage



.. table::

   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | Required Component                     | Notes                                                                                       |
   +========================================+=============================================================================================+
   | Linux x64                              |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |gcc link|_ or later                   |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |clang link|_                          | clang and python-clang bindings are needed for compile time code generation. The easiest    |
   |                                        | way to obtain them is 'pip install clang libclang'                                          |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |boost link|_ or later                 | Modules: *preprocessor*.                                                                    |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |cuda link|_                           |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |nvjpeg link|_                         | *This can be unofficially disabled. See below.*                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |protobuf link|_                       |  Supported version: 3.11.1                                                                  |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |cmake link|_ or later                 |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |jpegturbo link|_ or later             | *This can be unofficially disabled. See below.*                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |libtiff link|_ or later               | *This can be unofficially disabled. See below.*                                             |
   |                                        | Note: libtiff should be built with zlib support                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |ffmpeg link|_ or later                | We recommend using version 4.2.2 compiled following the *instructions below*.               |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |libsnd link|_ or later                | We recommend using version 1.0.28 compiled following the *instructions below*.              |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | |opencv link|_ or later                | Supported version: 4.3.0                                                                    |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | (Optional) |lmdb link|_ or later       |                                                                                             |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | (Optional) |gds link|_                 | Only libcufile is required for the build process, and the installed header needs to land    |
   |                                        | in `/usr/local/cuda/include` directory.                                                     |
   +----------------------------------------+---------------------------------------------------------------------------------------------+
   | One or more of the following Deep Learning frameworks:                                                                               |
   |      * |mxnet link|_ ``mxnet-cu90`` or later                                                                                         |
   |      * |pytorch link|_                                                                                                               |
   |      * |tf link|_ or later                                                                                                           |
   +----------------------------------------+---------------------------------------------------------------------------------------------+


.. note::

  TensorFlow installation is required to build the TensorFlow plugin for DALI.

.. note::

  Items marked *"unofficial"* are community contributions that are believed to work but not officially tested or maintained by NVIDIA.

.. note::

  This software uses the FFmpeg licensed code under the LGPLv2.1. Its source can be downloaded `from here`__.

  .. __: `ffmpeg link`_

  FFmpeg was compiled using the following command line:

  .. code-block:: bash

    ./configure \
    --prefix=/usr/local \
    --disable-static \
    --disable-all \
    --disable-autodetect \
    --disable-iconv \
    --enable-shared \
    --enable-avformat \
    --enable-avcodec \
    --enable-avfilter \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,avi \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes  && \
    make

.. note::

  This software uses the libsnd licensed under the LGPLv2.1. Its source can be downloaded `from here`__.

  .. __: `libsnd link`_

  libsnd was compiled using the following command line:

  .. code-block:: bash

    ./configure && make


Build DALI
^^^^^^^^^^

1. Get DALI source code:

.. code-block:: bash

  git clone --recursive https://github.com/NVIDIA/DALI
  cd DALI

2. Create a directory for CMake-generated Makefiles. This will be the directory, that DALI's built in.

.. code-block:: bash

  mkdir build
  cd build

3. Run CMake. For additional options you can pass to CMake, refer to :ref:`OptionalCmakeParamsAnchor`.

.. code-block:: bash

  cmake -D CMAKE_BUILD_TYPE=Release ..

4. Build. You can use ``-j`` option to execute it in several threads

.. code-block:: bash

  make -j"$(nproc)"

.. _PythonBindingsAnchor:

Install Python bindings
+++++++++++++++++++++++

In order to run DALI using Python API, you need to install Python bindings

.. code-block:: bash

    cd build
    pip install dali/python

.. note::

  Although you can create a wheel here by calling ``pip wheel dali/python``, we don't really recommend doing so. Such whl is not self-contained (doesn't have all the dependencies) and it will work only on the system where you built DALI bare-metal. To build a wheel that contains the dependencies and might be therefore used on other systems, follow :ref:`DockerBuilderAnchor`.

Verify the build (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Obtain test data
++++++++++++++++

.. _DALI_extra_link: https://github.com/NVIDIA/DALI_extra#nvidia-dali

You can verify the build by running GTest and Nose tests. To do so, you'll need DALI_extra repository, which contains test data. To download it follow `DALI_extra README <https://github.com/NVIDIA/DALI_extra#nvidia-dali>`_. Keep in mind, that you need git-lfs to properly clone DALI_extra repo. To install git-lfs, follow `this tutorial <https://github.com/git-lfs/git-lfs/wiki/Tutorial>`_.


Set test data path
++++++++++++++++++

DALI uses ``DALI_EXTRA_PATH`` environment variable to localize the test data. You can set it by invoking:

.. code-block:: bash

  $ export DALI_EXTRA_PATH=<path_to_DALI_extra>
  e.g. export DALI_EXTRA_PATH=/home/yourname/workspace/DALI_extra

Run tests
+++++++++

DALI tests consist of 2 parts: C++ (GTest) and Python (usually Nose, but that's not always true). To run the tests there are convenient targets for Make, that you can run after building finished

.. code-block:: bash

  cd <path_to_DALI>/build
  make check-gtest check-python

Building DALI using Clang (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  This build is experimental. It is neither maintained nor tested. It is not guaranteed to work.
  We recommend using GCC for production builds.


.. code-block:: bash

  cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang  ..
  make -j"$(nproc)"

.. _OptionalCmakeParamsAnchor:

Optional CMake build parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  ``BUILD_PYTHON`` - build Python bindings (default: ON)
-  ``BUILD_TEST`` - include building test suite (default: ON)
-  ``BUILD_BENCHMARK`` - include building benchmarks (default: ON)
-  ``BUILD_LMDB`` - build with support for LMDB (default: OFF)
-  ``BUILD_NVTX`` - build with NVTX profiling enabled (default: OFF)
-  ``BUILD_NVJPEG`` - build with ``nvJPEG`` support (default: ON)
-  ``BUILD_NVJPEG2K`` - build with ``nvJPEG2k`` support (default: OFF)
-  ``BUILD_LIBTIFF`` - build with ``libtiff`` support (default: ON)
-  ``BUILD_FFTS`` - build with ``ffts`` support (default: ON)
-  ``BUILD_LIBSND`` - build with libsnd support (default: ON)
-  ``BUILD_NVOF`` - build with ``NVIDIA OPTICAL FLOW SDK`` support (default: ON)
-  ``BUILD_NVDEC`` - build with ``NVIDIA NVDEC`` support (default: ON)
-  ``BUILD_NVML`` - build with ``NVIDIA Management Library`` (``NVML``) support (default: ON)
-  ``BUILD_CUFILE`` - build with ``GPU Direct Storage support`` support (default: ON)
-  ``VERBOSE_LOGS`` - enables verbose loging in DALI. (default: OFF)
-  ``WERROR`` - treat all build warnings as errors (default: OFF)
-  ``BUILD_WITH_ASAN`` - build with ASAN support (default: OFF). To run issue:
-  ``BUILD_DALI_NODEPS`` - disables support for third party libraries that are normally expected to be available in the system
-  ``LINK_DRIVER`` - enables direct linking with driver libraries or an appropriate stub instead of dlopen
   it in the runtime (removes the requirement to have clang-python bindings available to generate the stubs)

.. warning::

  Enabling this option effectively results in only the most basic parts of DALI to compile (C++ core and kernels libraries).
  It is useful when wanting to use DALI processing primitives (kernels) directly without the need to use DALI's executor infrastructure.

.. code-block:: bash

  LD_LIBRARY_PATH=. ASAN_OPTIONS=symbolize=1:protect_shadow_gap=0 ASAN_SYMBOLIZER_PATH=$(shell which llvm-symbolizer)
  LD_PRELOAD= *PATH_TO_LIB_ASAN* /libasan.so. *X* *PATH_TO_BINARY*

  Where *X* depends on used compiler version, for example GCC 7.x uses 4. Tested with GCC 7.4, CUDA 10.0
  and libasan.4. Any earlier version may not work.

-  ``DALI_BUILD_FLAVOR`` - Allow to specify custom name sufix (i.e. 'nightly') for nvidia-dali whl package
-  *(Unofficial)* ``BUILD_JPEG_TURBO`` - build with ``libjpeg-turbo`` (default: ON)
-  *(Unofficial)* ``BUILD_LIBTIFF`` - build with ``libtiff`` (default: ON)

.. note::

   DALI release packages are built with the options listed above set to ON and NVTX turned OFF.
   Testing is done with the same configuration.
   We ensure that DALI compiles with all of those options turned OFF, but there may exist
   cross-dependencies between some of those features.

Following CMake parameters could be helpful in setting the right paths:

.. |libjpeg-turbo_cmake link| replace:: **libjpeg CMake docs page**
.. _libjpeg-turbo_cmake link: https://cmake.org/cmake/help/v3.11/module/FindJPEG.html
.. |protobuf_cmake link| replace:: **protobuf CMake docs page**
.. _protobuf_cmake link: https://cmake.org/cmake/help/v3.11/module/FindProtobuf.html

* FFMPEG_ROOT_DIR - path to installed FFmpeg
* NVJPEG_ROOT_DIR - where nvJPEG can be found (from CUDA 10.0 it is shipped with the CUDA toolkit so this option is not needed there)
* libjpeg-turbo options can be obtained from |libjpeg-turbo_cmake link|_
* protobuf options can be obtained from |protobuf_cmake link|_


Cross-compiling DALI C++ API for aarch64 Linux (Docker)
-------------------------------------------------------

.. note::

  Support for aarch64 Linux platform is experimental. Some of the features are available only for
  x86-64 target and they are turned off in this build. There is no support for DALI Python library
  on aarch64 yet. Some Operators may not work as intended due to x86-64 specific implementations.

Setup
^^^^^
Download the JetPack 4.4 SDK for NVIDIA Jetson using the SDK Manager, following the instruction
provided here: https://developer.nvidia.com/embedded/jetpack.
Then select CUDA for the host. After download process has been completed move
``cuda-repo-ubuntu1804-10-2-local-10.2.89-440.40_1.0-1_amd64.deb`` and
``cuda-repo-cross-aarch64-10-2-local-10.2.89_1.0-1_all.deb`` from the download folder
to main DALI folder (they are required for cross build).

Build the aarch64 Linux Build Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    docker build -t nvidia/dali:tools_aarch64-linux -f docker/Dockerfile.cuda_aarch64.deps .
    docker build -t nvidia/dali:builder_aarch64-linux --build-arg "AARCH64_CUDA_TOOL_IMAGE_NAME=nvidia/dali:tools_aarch64-linux" -f docker/Dockerfile.build.aarch64-linux .

Compile
^^^^^^^
From the root of the DALI source tree

.. code-block:: bash

    docker run -v $(pwd):/dali nvidia/dali:builder_aarch64-linux

The relevant python wheel will be in ``dali_root_dir/wheelhouse``

Cross-compiling DALI C++ API for aarch64 QNX (Docker)
-----------------------------------------------------

.. note::

  Support for aarch64 QNX platform is experimental. Some of the features are available only for
  x86-64 target and they are turned off in this build. There is no support for DALI Python library
  on aarch64 yet. Some Operators may not work as intended due to x86-64 specific implementations.

Setup
^^^^^
After aquiring the QNX Toolchain, place it in a directory called ``qnx`` in the root of the DALI tree.
Then using the SDK Manager for NVIDIA DRIVE, select **QNX** as the *Target Operating System*
and select **DRIVE OS 5.1.0.0 SDK**.

In STEP 02 under **Download & Install Options**, select *Download Now. Install Later*.
and agree to the Terms and Conditions. Once downloaded move the **cuda-repo-cross-qnx**
debian package into the ``qnx`` directory you created in the DALI tree.

Build the aarch64 Build Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    docker build -t nvidia/dali:tools_aarch64-qnx -f docker/Dockerfile.cuda_qnx.deps .
    docker build -t nvidia/dali:builder_aarch64-qnx --build-arg "QNX_CUDA_TOOL_IMAGE_NAME=nvidia/dali:tools_aarch64-qnx" -f docker/Dockerfile.build.aarch64-qnx .

Compile
^^^^^^^
From the root of the DALI source tree

.. code-block:: bash

    docker run -v $(pwd):/dali nvidia/dali:builder_aarch64-qnx

The relevant Python wheel will be inside ``$(pwd)/wheelhouse``.
