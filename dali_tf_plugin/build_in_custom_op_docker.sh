#!/bin/bash

set -o xtrace
set -e

PYTHON_DIST_PACKAGES=( $(python -c "import site; print(site.getsitepackages()[0])") )
DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali"

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')

LAST_CONFIG_INDEX=$(python ../qa/setup_packages.py -n -u tensorflow-gpu --cuda ${CUDA_VERSION})

PREBUILT_DIR=/prebuilt
mkdir -p ${PREBUILT_DIR}

mkdir -p $PREBUILT_DIR

for i in `seq 0 $LAST_CONFIG_INDEX`; do
    INST=$(python ../qa/setup_packages.py -i $i -u tensorflow-gpu --cuda ${CUDA_VERSION})
    if [[ "${INST}" == *"nvidia-tensorflow"* ]]; then \
        continue
    fi
    echo "Building DALI TF plugin for TF version ${INST}"
    pip install ${INST} -f /pip-packages

    SUFFIX=$(echo $INST | sed 's/.*=\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
    LIB_PATH="${PREBUILT_DIR}/libdali_tf_${SUFFIX}.so"

    source ./build_dali_tf.sh "${LIB_PATH}"

    pip uninstall -y tensorflow-gpu
done
