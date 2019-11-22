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


# Create build directory for cmake and enter it
mkdir $SRC_DIR/build
cd $SRC_DIR/build

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

# Create 'gcc' symlink so nvcc can find it
ln -s $CC $BUILD_PREFIX/bin/gcc

# Force -std=c++14 in CXXFLAGS
export CXXFLAGS=${CXXFLAGS/-std=c++??/-std=c++14}

# For some reason `aligned_alloc` is present when we use compiler version 5.4.x
# Adding NO_ALIGNED_ALLOC definition for cutt
export CXXFLAGS="${CXXFLAGS} -DNO_ALIGNED_ALLOC"

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
      -DBUILD_NVTX=${BUILD_NVTX:-OFF}                     \
      -DBUILD_PYTHON=${BUILD_PYTHON:-ON}                  \
      -DBUILD_LMDB=${BUILD_LMDB:-ON}                      \
      -DBUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}          \
      -DBUILD_NVJPEG=${BUILD_NVJPEG:-ON}                  \
      -DBUILD_LIBTIFF=${BUILD_LIBTIFF:-ON}                \
      -DBUILD_NVOF=${BUILD_NVOF:-ON}                      \
      -DBUILD_NVDEC=${BUILD_NVDEC:-ON}                    \
      -DBUILD_LIBSND=${BUILD_LIBSND:-ON}                  \
      -DBUILD_NVML=${BUILD_NVML:-ON}                      \
      -DBUILD_FFTS=${BUILD_FFTS:-ON}                      \
      -DVERBOSE_LOGS=${VERBOSE_LOGS:-OFF}                 \
      -DWERROR=${WERROR:-ON}                              \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}           \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}     \
      -DTIMESTAMP=${DALI_TIMESTAMP} -DGIT_SHA=${GIT_SHA-${GIT_FULL_HASH}} \
      ..
make -j"$(nproc --all)"
make install

# set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN$UPDIRS, $ORIGIN$UPDIRS/.libs
PKGNAME_PATH=$PWD/dali/python/nvidia/dali
find $PKGNAME_PATH -type f -name "*.so*" -o -name "*.bin" | while read FILE; do
    UPDIRS=$(dirname $(echo "$FILE" | sed "s|$PKGNAME_PATH||") | sed 's/[^\/][^\/]*/../g')
    echo "Setting rpath of $FILE to '\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs'"
    patchelf --set-rpath "\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs" $FILE
    patchelf --print-rpath $FILE
done

# pip install
$PYTHON -m pip install --no-deps --ignore-installed -v dali/python

# Build tensorflow plugin
export LD_LIBRARY_PATH="$PREFIX/libjpeg-turbo/lib:$PREFIX/lib:$LD_LIBRARY_PATH"
DALI_PATH=$($PYTHON -c 'import nvidia.dali as dali; import os; print(os.path.dirname(dali.__file__))')
echo "DALI_PATH is ${DALI_PATH}"
pushd $SRC_DIR/dali_tf_plugin/
source ./build_dali_tf.sh $DALI_PATH/plugin/libdali_tf_current.so
popd

# Move tfrecord2idx to host env so it can be found at runtime
cp $SRC_DIR/tools/tfrecord2idx $PREFIX/bin
