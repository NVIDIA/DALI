#!/bin/bash -e

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/../..
source $topdir/qa/setup_test_common.sh

# save old CUDA symlink, remove CUDA wheel that is suppose to be latest
version_eq "$CUDA_VERSION_MAJOR" "11" && \
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

pushd ../TL0_python-self-test-operators_1
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-operators_2
bash -e ./test.sh
popd

# restore old CUDA symlink, reinstall the latest CUDA wheels
version_eq "$CUDA_VERSION_MAJOR" "11" && \
  rm -rf /usr/local/cuda && mv /usr/local/cuda_bak /usr/local/cuda && \
  pip install nvidia-cufft-cu${DALI_CUDA_MAJOR_VERSION}  \
              nvidia-npp-cu${DALI_CUDA_MAJOR_VERSION}    \
              nvidia-nvjpeg-cu${DALI_CUDA_MAJOR_VERSION} \
  || true
