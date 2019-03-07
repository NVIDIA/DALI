#!/bin/bash -e
# used pip packages
pip_packages="jupyter matplotlib opencv-python mxnet-cu##CUDA_VERSION## tensorflow-gpu torchvision torch"

# We need cmake to run the custom plugin notebook + ffmpeg and wget for video example
apt-get update
apt-get install -y --no-install-recommends wget ffmpeg cmake

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

cd docs/examples

test_body() {
    # dummy patern
    black_list_files="#"
    # black list TensorFlow releated tests for CUDA 10.0 as TF is not supporting CUDA 10.0
    if [ "${CUDA_VERSION}" == "100" ]; then
       black_list_files="tensorflow\|"${black_list_files}
    fi

    # test code
    find */* -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
    python${PYVER:0:1} pytorch/resnet50/main.py -t
}

source ../../qa/test_template.sh

popd
