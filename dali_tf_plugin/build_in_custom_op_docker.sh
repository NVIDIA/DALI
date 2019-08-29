#!/bin/bash

set -o xtrace
set -e

PYTHON_DIST_PACKAGES=( $(python -c "import site; print(site.getsitepackages()[0])") )
DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali"

CUDA_VERSION=$(cat /usr/local/cuda/version.txt | head -1 | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
LAST_CONFIG_INDEX=$(python ../qa/setup_packages.py -n -u tensorflow-gpu --cuda ${CUDA_VERSION})
for i in `seq 0 $LAST_CONFIG_INDEX`;
do
    INST=$(python ../qa/setup_packages.py -i $i -u tensorflow-gpu --cuda ${CUDA_VERSION})
    echo "Building DALI TF plugin for TF version ${INST}"
    pip install ${INST} -f /pip-packages

    SUFFIX=$(echo $INST | sed 's/.*=\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/')
    LIB_NAME=libdali_tf_${SUFFIX}.so

    source ./build_dali_tf.sh "${LIB_NAME}"

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
