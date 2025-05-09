#!/bin/bash
#
# (C) Copyright IBM Corp. 2019. All Rights Reserved.
# (C) Copyright NVIDIA CORPORATION. 2019. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#Determine Architecture

ARCH="$(arch)"
if [ ${ARCH} = "x86_64" ]; then
    ARCH_LONGNAME="x86_64-conda_cos6"
elif [ ${ARCH} = "ppc64le" ]; then
    ARCH_LONGNAME="powerpc64le-conda_cos7"
else
    echo "Error: Unsupported Architecture. Expected: [x86_64|ppc64le] Actual: ${ARCH}"
    exit 1
fi

# Create 'gcc' and 'g++' symlinks so nvcc can find it
ln -s $CC $BUILD_PREFIX/bin/gcc
ln -s $CXX $BUILD_PREFIX/bin/g++

# Force -std=c++17 in CXXFLAGS
export CXXFLAGS=${CXXFLAGS/-std=c++??/-std=c++17}

# For some reason `aligned_alloc` is present when we use compiler version 5.4.x
# Adding NO_ALIGNED_ALLOC definition for cutt
export CXXFLAGS="${CXXFLAGS} -DNO_ALIGNED_ALLOC"
export PATH=/usr/local/cuda/bin:${PATH}

# make it on by default for CUDA 11.x
if [ "${CUDA_VERSION/./}" -ge 110 ]; then
  export WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT=ON
else
  export WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT=OFF
fi


export BUILD_NVCOMP=${BUILD_NVCOMP:-OFF}

# Create build directory for cmake and enter it
mkdir $SRC_DIR/build_bindings
cd $SRC_DIR/build_bindings

# allow DALI import all dependencies in the build env
export LD_LIBRARY_PATH="$PREFIX/libjpeg-turbo/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# Build
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCUDA_rt_LIBRARY=$BUILD_PREFIX/${ARCH_LONGNAME}-linux-gnu/sysroot/usr/lib/librt.so \
      -DCUDA_CUDA_LIBRARY=/usr/local/cuda/targets/${ARCH}-linux/lib/stubs/libcuda.so \
      -DCUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}            \
      -DNVJPEG_ROOT_DIR=/usr/local/cuda                   \
      -DFFMPEG_ROOT_DIR=$PREFIX/lib                       \
      -DCMAKE_PREFIX_PATH="$PREFIX/libjpeg-turbo;$PREFIX" \
      -DCMAKE_INSTALL_PREFIX=$PREFIX                      \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}     \
      -DBUILD_TEST=${BUILD_TEST:-ON}                      \
      -DBUILD_BENCHMARK=${BUILD_BENCHMARK:-ON}            \
      -DBUILD_NVTX=${BUILD_NVTX}                          \
      -DBUILD_PYTHON=ON                                   \
      -DPREBUILD_DALI_LIBS=ON                             \
      -DPYTHON_STUBGEN_INTERPRETER=${PYTHON}              \
      -DBUILD_LMDB=${BUILD_LMDB:-ON}                      \
      -DBUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}          \
      -DBUILD_OPENCV=${BUILD_OPENCV:-ON}                  \
      -DBUILD_PROTOBUF=${BUILD_PROTOBUF:-ON}              \
      -DBUILD_NVJPEG=${BUILD_NVJPEG:-ON}                  \
      -DBUILD_NVJPEG2K=${BUILD_NVJPEG2K:-ON}              \
      -DBUILD_LIBTIFF=${BUILD_LIBTIFF:-ON}                \
      -DBUILD_LIBSND=${BUILD_LIBSND:-ON}                  \
      -DBUILD_LIBTAR=${BUILD_LIBTAR:-ON}                  \
      -DBUILD_FFTS=${BUILD_FFTS:-ON}                      \
      -DBUILD_NVOF=${BUILD_NVOF:-ON}                      \
      -DBUILD_NVDEC=${BUILD_NVDEC:-ON}                    \
      -DBUILD_NVML=${BUILD_NVML:-ON}                      \
      -DBUILD_CUFILE=${BUILD_CUFILE:-ON}                  \
      -DBUILD_NVCOMP=${BUILD_NVCOMP:-ON}                  \
      -DBUILD_CVCUDA=${BUILD_CVCUDA:-ON}                  \
      -DBUILD_NVIMAGECODEC=${BUILD_NVIMAGECODEC:-ON}      \
      -DLINK_LIBCUDA=${LINK_LIBCUDA:-OFF}                 \
      -DWITH_DYNAMIC_CUDA_TOOLKIT=${WITH_DYNAMIC_CUDA_TOOLKIT:-${WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT}}\
      -DWITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG:-ON}    \
      -DWITH_DYNAMIC_CUFFT=${WITH_DYNAMIC_CUFFT:-ON}      \
      -DWITH_DYNAMIC_NPP=${WITH_DYNAMIC_NPP:-ON}          \
      -DWITH_DYNAMIC_NVIMGCODEC=${WITH_DYNAMIC_NVIMGCODEC:-ON} \
      -DWITH_DYNAMIC_NVCOMP=${WITH_DYNAMIC_NVCOMP:-ON  }  \
      -DVERBOSE_LOGS=${VERBOSE_LOGS:-OFF}                 \
      -DWERROR=${WERROR:-ON}                              \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}           \
      -DBUILD_WITH_LSAN=${BUILD_WITH_LSAN:-OFF}           \
      -DBUILD_WITH_UBSAN=${BUILD_WITH_UBSAN:-OFF}         \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}     \
      -DTIMESTAMP=${DALI_TIMESTAMP} -DGIT_SHA=${GIT_SHA-${GIT_FULL_HASH}} \
      ..
make -j"$(nproc --all)" dali_python python_function_plugin copy_post_build_target dali_python_generate_stubs install_headers

# pip install
$PYTHON -m pip install --no-deps --ignore-installed -v dali/python

DALI_PATH=$($PYTHON -c 'import nvidia.dali as dali; import os; print(os.path.dirname(dali.__file__))')
echo "DALI_PATH is ${DALI_PATH}"

# Move tfrecord2idx to host env so it can be found at runtime
cp $SRC_DIR/tools/tfrecord2idx $PREFIX/bin
