#!/bin/bash -e

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/../..
source $topdir/qa/setup_test_common.sh

# save old CUDA symlink, remove CUDA wheel that is suppose to be latest
version_ge "$CUDA_VERSION" "110" && \
  mv /usr/local/cuda /usr/local/cuda_bak && \
  ln -s cuda-11.1 /usr/local/cuda && \
  pip uninstall -y `pip list | grep nvidia-cu | cut -d " " -f1` `pip list | grep nvidia-n | cut -d " " -f1` \
  || true

pushd ../TL0_python-self-test-core
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-readers-decoders
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-operators
bash -e ./test.sh
popd

# restore old CUDA symlink, reinstall the latest CUDA wheels
version_ge "$CUDA_VERSION" "110" && \
  rm -rf /usr/local/cuda && mv /usr/local/cuda_bak /usr/local/cuda && \
  pip install nvidia-cufft-cu11 nvidia-npp-cu11 nvidia-nvjpeg-cu11 \
  || true
