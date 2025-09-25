#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} dataclasses numpy opencv-python pillow librosa scipy nvidia-ml-py==11.450.51 numba lz4 numba_cuda[cu${DALI_CUDA_MAJOR_VERSION}]>0.19.0'

target_dir=./dali/test/python

test_body() {
  if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
    exit 0
  fi
  # make sure nvcc and nvjitlink are the same version, if the nvjitlink is lower than nvcc
  # we get the PTX version mismatch error
  if [[ "$DALI_CUDA_MAJOR_VERSION" == "12" ]]; then
    nvcc_pkg="nvidia-cuda-nvcc-cu12"
    nvjitlink_pkg="nvidia-nvjitlink-cu12"
  else
    # with CUDA 13 cuXX suffix is not used and CUDA version is defined by the major version
    nvcc_pkg="nvidia-cuda-nvcc"
    nvjitlink_pkg="nvidia-nvjitlink"
  fi
  nvcc_ver=$(pip show $nvcc_pkg | awk '/^Version: /{print $2}') || true
  nvjitlink_ver=$(pip show $nvjitlink_pkg | awk '/^Version: /{print $2}') || true
  if [[ -n "$nvcc_ver" && -n "$nvjitlink_ver" ]]; then
    nvcc_majmin=$(echo $nvcc_ver | awk -F. '{print $1"."$2}')
    nvjitlink_majmin=$(echo $nvjitlink_ver | awk -F. '{print $1"."$2}')
    if [[ "$nvcc_majmin" != "$nvjitlink_majmin" ]]; then
      awk_comp=$(awk -v a="$nvcc_majmin" -v b="$nvjitlink_majmin" 'BEGIN{print (a<b)?1:0}')
      if [[ "$awk_comp" == "1" ]]; then
        pip install --upgrade "${nvcc_pkg}~=$nvjitlink_majmin"
      else
        pip install --upgrade "${nvjitlink_pkg}~=$nvcc_majmin"
      fi
    fi
  fi
  ${python_new_invoke_test} -A '!slow' -s operator_1 test_numba_func
}

pushd ../..
source ./qa/test_template.sh
popd
