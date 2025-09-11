#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy matplotlib pillow opencv-python-headless librosa simpleaudio'
target_dir=./docs/examples

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

do_once() {
  # We need cmake to run the custom plugin notebook + ffmpeg, wget for video example, libasound2-dev for audio test
  # install native compilers in conda instead of using system ones so we can link with conda packages
  enable_conda
  # Pinning cmake to <= 4.0.3 due to strange interactions of later versions with Jupyter
  conda install gcc==9.4 gxx==9.4 alsa-lib wget ffmpeg cmake -y
  mkdir -p idx_files
  disable_conda
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
