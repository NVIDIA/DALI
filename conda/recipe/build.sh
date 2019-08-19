#!/bin/bash
#
# (C) Copyright IBM Corp. 2019. All Rights Reserved.
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
ln -s $CONDA_PREFIX/bin/${ARCH_LONGNAME}-linux-gnu-gcc $CONDA_PREFIX/bin/gcc

# 1. Conda environment is using -std=c++17 which seems to override our settings in CMakeLists.txt
#    Because of that, we are replacing c++17 with c++14 here
export CXXFLAGS="${CXXFLAGS/-std=c++17/-std=c++14}"

# Build
# BUILD_TENSORFLOW No longer exists. Previous flag to build tf plugin (used in release_v0.9)
cmake -DBUILD_TENSORFLOW=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCUDA_rt_LIBRARY=$CONDA_PREFIX/${ARCH}-linux-gnu/sysroot/usr/lib/librt.so \
      -DNVJPEG_ROOT_DIR=$CONDA_PREFIX/lib64 \
      -DFFMPEG_ROOT_DIR=$CONDA_PREFIX/lib \
      -DCMAKE_PREFIX_PATH="$CONDA_PREFIX/libjpeg-turbo;$CONDA_PREFIX" \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DCUDA_CUDA_LIBRARY=/usr/local/cuda/targets/${ARCH_SHORTNAME}-linux/lib/stubs/libcuda.so \
      ..

make VERBOSE=1 -j"$(nproc --all)" install

export PYTHONUSERBASE=$PREFIX

$CONDA_PREFIX/bin/pip install --user dali/python

# Move required .so files to $PREFIX/lib
cp $SRC_DIR/build/dali/python/nvidia/dali/*.so $PREFIX/lib

# Build DALI TF plugin
pushd $SRC_DIR/dali_tf_plugin/
source ./build_dali_tf.sh $PREFIX/lib/libdali_tf_current.so
popd

# Move tfrecord2idx to host env so it can be found at runtime
cp $SRC_DIR/tools/tfrecord2idx $PREFIX/bin
