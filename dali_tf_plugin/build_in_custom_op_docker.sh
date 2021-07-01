#!/bin/bash

# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -o xtrace
set -e

PYTHON_DIST_PACKAGES=( $(python -c "import site; print(site.getsitepackages()[0])") )
DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali"

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')

LAST_CONFIG_INDEX=$(python ../qa/setup_packages.py -n -u tensorflow-gpu --cuda ${CUDA_VERSION})

PREBUILT_DIR=/prebuilt
mkdir -p ${PREBUILT_DIR}

STUB_DIR="${PREBUILT_DIR}/stub"
STUB_LIB="${STUB_DIR}/libdali.so"
mkdir -p "${STUB_DIR}"
source ./build_dali_stub.sh "${STUB_LIB}"

for i in `seq 0 $LAST_CONFIG_INDEX`; do
    INST=$(python ../qa/setup_packages.py -i $i -u tensorflow-gpu --cuda ${CUDA_VERSION})
    if [[ "${INST}" == *"nvidia-tensorflow"* ]]; then \
        continue
    fi
    echo "Building DALI TF plugin for TF version ${INST}"
    pip install ${INST} -f /pip-packages

    SUFFIX=$(echo $INST | sed 's/.*=\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
    LIB_PATH="${PREBUILT_DIR}/libdali_tf_${SUFFIX}.so"

    source ./build_dali_tf.sh "${LIB_PATH}" "${STUB_DIR}"

    pip uninstall -y tensorflow-gpu
done
