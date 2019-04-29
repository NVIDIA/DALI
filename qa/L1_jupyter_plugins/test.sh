#!/bin/bash -e

source ../setup_test.sh

# used pip packages
pip_packages="jupyter matplotlib mxnet-cu##CUDA_VERSION## tensorflow-gpu torchvision torch"

# We need cmake to run the custom plugin notebook + ffmpeg and wget for video example
apt-get update
apt-get install -y --no-install-recommends wget ffmpeg cmake

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

# Apparently gcc/g++ installation is broken in the docker image
if ( ! test `find /usr/lib/gcc -name stddef.h` ); then
    apt-get purge --autoremove -y build-essential g++ gcc libc6-dev
    apt-get update && apt-get install -y build-essential g++ gcc libc6-dev
fi

cd docs/examples

test_body() {
    black_list_files="optical_flow_example.ipynb\|#" # optical flow requires TU102 architecture
                                                     # whilst currently L1_jupyter_plugins test
                                                     # can be run only on V100
                                                     # TensorFlow doesn't support Python 3.7 yet
    if [ $PYTHON_VERSION == "3.7" ]; then
        black_list_files="tensorflow\|$black_list_files"
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
