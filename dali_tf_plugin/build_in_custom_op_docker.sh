#!/bin/bash

set -o xtrace
set -e

SRCS="daliop.cc"

INCL_DIRS="-I/usr/local/cuda/include/"

PYTHON_DIST_PACKAGES=( $(python -c "import site; print(site.getsitepackages()[0])") )
DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali"

DALI_CFLAGS=( $(python ./dali_compile_flags.py --cflags) )
DALI_LFLAGS=( $(python ./dali_compile_flags.py --lflags) )

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | head -1 | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1/')
test ${CUDA_VERSION} = "9"  && export SUPPORTED_TF_VERSIONS="1.7.0 1.11.0 1.12.0"
test ${CUDA_VERSION} = "10" && export SUPPORTED_TF_VERSIONS="1.13.1 1.14.0"

for TF_VERSION in ${SUPPORTED_TF_VERSIONS}; do
    echo "Building DALI TF plugin for TF version ${TF_VERSION}"
    pip install tensorflow-gpu=="${TF_VERSION}"

    SUFFIX=$(echo $TF_VERSION | sed 's/\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
    LIB_NAME=libdali_tf_${SUFFIX}.so
    TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
    TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

    g++ -std=c++11 -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} ${DALI_CFLAGS[@]} ${DALI_LFLAGS[@]}

    pip uninstall -y tensorflow-gpu
done

mkdir -p dali_tf_sdist_build
cd dali_tf_sdist_build

cmake .. \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} \
      -DGIT_SHA=${GIT_SHA}

make -j
python setup.py sdist
cp dist/*.tar.gz /dali_tf_sdist
