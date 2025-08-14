#!/bin/bash -e

topdir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/../..
source $topdir/qa/setup_test_common.sh

# save old CUDA symlink, remove CUDA wheel that is suppose to be latest
version_eq "$DALI_CUDA_MAJOR_VERSION" "11" && \
  mv /usr/local/cuda /usr/local/cuda_bak && \
  ln -s cuda-11.2 /usr/local/cuda
version_eq "$DALI_CUDA_MAJOR_VERSION" "12" && \
  mv /usr/local/cuda /usr/local/cuda_bak && \
  ln -s cuda-12.0 /usr/local/cuda
version_eq "$DALI_CUDA_MAJOR_VERSION" "13" && \
  mv /usr/local/cuda /usr/local/cuda_bak && \
  ln -s cuda-13.0 /usr/local/cuda
version_ge "$DALI_CUDA_MAJOR_VERSION" "11" && \
  pip uninstall -y `pip list | grep nvidia-cufft | cut -d " " -f1` \
                   `pip list | grep nvidia-nvjpeg | cut -d " " -f1` \
                   `pip list | grep nvidia-nvjpeg2k | cut -d " " -f1` \
                   `pip list | grep nvidia-nvtiff | cut -d " " -f1` \
                   `pip list | grep nvidia-npp | cut -d " " -f1` \
  || true

export DO_NOT_INSTALL_CUDA_WHEEL="TRUE"

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
version_eq "$DALI_CUDA_MAJOR_VERSION" "11" && \
  rm -rf /usr/local/cuda && mv /usr/local/cuda_bak /usr/local/cuda
version_ge "$DALI_CUDA_MAJOR_VERSION" "11" && \
  if [ "$DALI_CUDA_MAJOR_VERSION" -ge 13 ]; then
      pip install --upgrade nvidia-npp~=${DALI_CUDA_MAJOR_VERSION}.0    \
                            nvidia-nvjpeg~=${DALI_CUDA_MAJOR_VERSION}.0 \
                            nvidia-cufft~=$((DALI_CUDA_MAJOR_VERSION-1)).0  \
                            nvidia-nvjpeg2k-cu${DALI_CUDA_MAJOR_VERSION} \
                            nvidia-nvtiff-cu${DALI_CUDA_MAJOR_VERSION}
  else
      pip install --upgrade nvidia-npp-cu${DALI_CUDA_MAJOR_VERSION}${NPP_VERSION} \
                            nvidia-nvjpeg-cu${DALI_CUDA_MAJOR_VERSION} \
                            nvidia-nvjpeg2k-cu${DALI_CUDA_MAJOR_VERSION} \
                            nvidia-nvtiff-cu${DALI_CUDA_MAJOR_VERSION} \
                            nvidia-cufft-cu${DALI_CUDA_MAJOR_VERSION}
  fi
