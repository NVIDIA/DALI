#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib pillow opencv-python librosa"
if [ "$PYVER" != "2.7" ]; then
    pip_packages="${pip_packages} simpleaudio" 
fi
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
    black_list_files="multigpu\|mxnet.*\|tensorflow.*\|pytorch.*\|paddle*\|optical_flow.*\|python_operator.*\|#"

    # Blacklist for python2. Can be removed after dropping python2
    if [ "$PYVER" == "2.7" ]; then
        black_list_files="audio_decoder\|"${black_list_files}
    fi

    find * -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

pushd ../..
source ./qa/test_template.sh
popd
