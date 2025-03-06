#!/bin/bash

# Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

COMPILER=${CXX:-g++}
PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
DALI_STUB_DIR=${2:-"${PWD}/stub"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/"

DALI_CFLAGS="-I../include -I.."
DALI_LFLAGS="-L${DALI_STUB_DIR} -ldali"

TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_VERSION=( $($PYTHON -c 'import tensorflow as tf; print(tf.__version__)') )
TF_VERSION=$(echo $TF_VERSION | tr "." "\n")
TF_VERSION_MAJOR=$(echo $TF_VERSION | cut -d' ' -f1)
TF_VERSION_MINOR=$(echo $TF_VERSION | cut -d' ' -f2)
TF_VERSION_PATCH=$(echo $TF_VERSION | cut -d' ' -f3)

CPP_VER=( $($PYTHON -c "import tensorflow as tf; from packaging.version import Version; print('--std=c++14' if Version(tf.__version__) < Version('2.10') else '--std=c++17')") )

# Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
# https://github.com/tensorflow/tensorflow/issues/17316
# Do not remove it.
$COMPILER -Wl,-rpath,\$ORIGIN ${CPP_VER} -DNDEBUG -O2 -shared -fPIC ${SRCS} \
    -o ${LIB_NAME} ${INCL_DIRS} ${DALI_CFLAGS} ${DALI_LFLAGS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -DTF_MAJOR_VERSION=${TF_VERSION_MAJOR} -DTF_MINOR_VERSION=${TF_VERSION_MINOR} -DTF_PATCH_VERSION=${TF_VERSION_PATCH}
