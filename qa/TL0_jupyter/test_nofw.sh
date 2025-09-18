#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy matplotlib pillow opencv-python-headless librosa simpleaudio'
target_dir=./docs/examples

do_once() {
  # We need cmake to run the custom plugin notebook + ffmpeg, wget for video example, libasound2-dev for audio test
  apt-get update
  apt-get install -y --no-install-recommends wget ffmpeg libasound2-dev
  mkdir -p idx_files
  wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-Linux-x86_64.sh
  bash cmake-*.sh --skip-license --prefix=/usr
  rm cmake-*.sh
}

test_body() {
    # test code
    # test all jupyters except one related to a particular FW,
    # and one requiring a dedicated HW (multiGPU, GDS and OF)
    # optical flow requires TU102 architecture whilst this test can be run on any GPU
    exclude_files="multigpu\|mxnet\|tensorflow\|pytorch\|paddle\|jax\|external_input.ipynb\|numpy_reader.ipynb\|webdataset-externalsource.ipynb\|optical_flow\|python_operator\|#"

    find * -name "*.ipynb" | sed "/${exclude_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

pushd ../..
source ./qa/test_template.sh
popd
