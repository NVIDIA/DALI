#!/bin/bash

set -o xtrace

pip install whl/*.whl

SRCS="daliop.cc"

SUFFIX=$(echo $TF_VERSION | sed 's/\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
LIB_NAME=libdali_tf_${SUFFIX}.so

INCL_DIRS="-I/usr/local/cuda/targets/x86_64-linux/include/"

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

PYTHON_DIST_PACKAGES=( $(python -c "import site; print(site.getsitepackages()[0])") )
DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali"

# Requires CUDA
# DALI_CFLAGS=( $(python -c 'import nvidia.dali as dali; print(" ".join(dali.sysconfig.get_compile_flags()))') )
DALI_CFLAGS="-I${DALI_TOPDIR}/include -D_GLIBCXX_USE_CXX11_ABI=0"

# Requires CUDA
# DALI_LFLAGS=( $(python -c 'import nvidia.dali as dali; print(" ".join(dali.sysconfig.get_link_flags()))') )
DALI_LFLAGS="-L${DALI_TOPDIR} -ldali"

g++ -std=c++11 -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} ${DALI_CFLAGS[@]} ${DALI_LFLAGS[@]}
