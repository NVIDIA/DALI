#!/bin/bash

# Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
DALI_STUB_DIR=${2:-"${PWD}/stub"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/"

DALI_CFLAGS="-I../include -I.."
DALI_LFLAGS="-L${DALI_STUB_DIR} -ldali"

TF_CONFIG=$($PYTHON - <<'PY'
import shlex
import tensorflow as tf
from packaging.version import Version
from dali_tf_plugin_utils import get_cpp_compiler

tf_cflags = tf.sysconfig.get_compile_flags()
tf_lflags = tf.sysconfig.get_link_flags()
compiler = get_cpp_compiler(" ".join(tf_cflags))
cpp_ver = "--std=c++14" if Version(tf.__version__) < Version("2.10") else "--std=c++17"
major, minor, patch = tf.__version__.split(".")[:3]

# Quote every value before emitting shell assignments.
print(f"TF_CFLAGS=({shlex.join(tf_cflags)})")
print(f"TF_LFLAGS=({shlex.join(tf_lflags)})")
for name, value in (
    ("COMPILER", compiler),
    ("CPP_VER", cpp_ver),
    ("TF_VERSION_MAJOR", major),
    ("TF_VERSION_MINOR", minor),
    ("TF_VERSION_PATCH", patch),
):
    print(f"{name}={shlex.quote(value)}")
PY
) || exit 1
eval "$TF_CONFIG"

# Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
# https://github.com/tensorflow/tensorflow/issues/17316
# Do not remove it.
"$COMPILER" -Wl,-rpath,\$ORIGIN ${CPP_VER} -DNDEBUG -O2 -shared -fPIC ${SRCS} \
    -o ${LIB_NAME} ${INCL_DIRS} ${DALI_CFLAGS} ${DALI_LFLAGS} "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" \
    -DTF_MAJOR_VERSION=${TF_VERSION_MAJOR} -DTF_MINOR_VERSION=${TF_VERSION_MINOR} -DTF_PATCH_VERSION=${TF_VERSION_PATCH}
