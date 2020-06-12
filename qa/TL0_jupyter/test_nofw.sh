#!/bin/bash -e
# used pip packages
# lock numba version as 0.50 changed module location and librosa hasn't catched up in 7.2 yet
pip_packages="jupyter numpy matplotlib pillow opencv-python librosa simpleaudio numba==0.49"
target_dir=./docs/examples

do_once() {
  # We need cmake to run the custom plugin notebook + ffmpeg, wget for video example, libasound2-dev for audio test
  apt-get update
  apt-get install -y --no-install-recommends wget ffmpeg cmake libasound2-dev
  mkdir -p idx_files
}

test_body() {
    # test code
    # test all jupyters except one related to a particular FW,
    # and one requiring a dedicated HW (multiGPU and OF)
    # optical flow requires TU102 architecture whilst this test can be run on any GPU
    black_list_files="multigpu\|mxnet*\|tensorflow*\|pytorch*\|paddle*\|optical_flow*\|python_operator*\|#"

    find * -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

pushd ../..
source ./qa/test_template.sh
popd
