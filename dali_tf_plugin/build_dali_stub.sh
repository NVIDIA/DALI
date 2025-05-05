#!/bin/bash

# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
OUT_DALI_STUB_LIB=${1:-"${PWD}/stub/libdali.so"}
INCL_DIRS="-I/usr/local/cuda/include/"

DALI_STUB_DIR=`mktemp -d -t "dali_stub_XXXXXX"`
DALI_STUB_SRC="${DALI_STUB_DIR}/dali_stub.cc"
$PYTHON ../tools/stubgen.py ../include/dali/dali.h --output "${DALI_STUB_SRC}"

DALI_CFLAGS="-I../include -I.."
$COMPILER -std=c++14 -DNDEBUG -O2 -shared -fPIC ${DALI_STUB_SRC} -o ${OUT_DALI_STUB_LIB} ${INCL_DIRS} ${DALI_CFLAGS}

# Cleaning up stub dir
rm -rf "${DALI_STUB_DIR}"
