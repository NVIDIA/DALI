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

# Force -std=c++14 in CXXFLAGS
export CXXFLAGS=${CXXFLAGS/-std=c++??/-std=c++14}

# For some reason `aligned_alloc` is present when we use compiler version 5.4.x
# Adding NO_ALIGNED_ALLOC definition for cutt
export CXXFLAGS="${CXXFLAGS} -DNO_ALIGNED_ALLOC"
export PATH=/usr/local/cuda/bin:${PATH}

# For some reason `aligned_alloc` is present when we use compiler version 5.4.x
# Adding NO_ALIGNED_ALLOC definition for cutt
export CXXFLAGS="${CXXFLAGS} -DNO_ALIGNED_ALLOC"
export PATH=/usr/local/cuda/bin:${PATH}

# Create build directory for cmake and enter it
mkdir $SRC_DIR/build
cd $SRC_DIR/build
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

# bundle FFmpeg to make sure DALI ships and uses own version
fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    BASENAME=$(basename $1)
    INITNAME=$(echo $BASENAME | cut -f1 -d".")
    ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
    echo "$INITNAME-$HASH.$ENDNAME"
}

DEPS_LIST=(
    "$PREFIX/lib/libavformat.so.58"
    "$PREFIX/lib/libavcodec.so.58"
    "$PREFIX/lib/libavfilter.so.7"
    "$PREFIX/lib/libavutil.so.56"
)

DEPS_SONAME=(
    "libavformat.so.58"
    "libavcodec.so.58"
    "libavfilter.so.7"
    "libavutil.so.56"
)

PKGNAME_PATH=dali/python/nvidia/dali/
mkdir -p $PKGNAME_PATH/.libs

# use LIEF master until a release with perf optimization is available
TMP_PATH="$(mktemp -d)"
pushd $TMP_PATH

git clone https://github.com/lief-project/LIEF
cd LIEF

mkdir build
cd build

cmake ../
make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
cp api/python/lief.so $(python -c "import sys; print(sys.path[-1])")
rm -rf $TMP_PATH

popd

patch_elf="python ../dali/python/patcher.py"
patched=()
for filepath in "${DEPS_LIST[@]}"; do
    filename=$(basename $filepath)
    patchedname=$(fname_with_sha256 $filepath)
    patchedpath=$PKGNAME_PATH/.libs/$patchedname
    patched+=("$patchedname")

    if [[ ! -f "$filepath" ]]; then
        echo "Didn't find $filename, skipping..."
        continue
    fi
    echo "Copying $filepath to $patchedpath"
    cp $filepath $patchedpath

    echo "Patching DT_SONAME field in $patchedpath"
    ${patch_elf} --set-soname $patchedname $patchedpath
done

find $PKGNAME_PATH -name '*.so*' -o -name '*.bin' | while read sofile; do
    echo "*************************"
    echo "patching: $sofile"
    echo "from:  ${DEPS_SONAME[@]//[$'\t\r\n']}"
    echo "to:    ${patched[@]//[$'\t\r\n']}"
    ${patch_elf} --replace-needed ${DEPS_SONAME} ${patched} $sofile
    echo "*************************"
done

# set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN$UPDIRS, $ORIGIN$UPDIRS/.libs
PKGNAME_PATH=$PWD/dali/python/nvidia/dali
find $PKGNAME_PATH -type f -name "*.so*" -o -name "*.bin" | while read FILE; do
    UPDIRS=$(dirname $(echo "$FILE" | sed "s|$PKGNAME_PATH||") | sed 's/[^\/][^\/]*/../g')
    echo "Setting rpath of $FILE to '\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs'"
    ${patch_elf}--set-rpath "\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs" $FILE
    ${patch_elf} --print-rpath $FILE
done

# pip install
$PYTHON -m pip install --no-deps --ignore-installed -v dali/python

# Build tensorflow plugin
export LD_LIBRARY_PATH="$PREFIX/libjpeg-turbo/lib:$PREFIX/lib:$LD_LIBRARY_PATH"
DALI_PATH=$($PYTHON -c 'import nvidia.dali as dali; import os; print(os.path.dirname(dali.__file__))')
echo "DALI_PATH is ${DALI_PATH}"
pushd $SRC_DIR/dali_tf_plugin/
mkdir -p dali_tf_sdist_build
cd dali_tf_sdist_build

cmake .. \
      -DCUDA_VERSION:STRING="${CUDA_VERSION}" \
      -DDALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR} \
      -DTIMESTAMP=${DALI_TIMESTAMP} \
      -DGIT_SHA=${GIT_SHA}
make -j install
$PYTHON -m pip install --no-deps --ignore-installed .
popd

# Move tfrecord2idx to host env so it can be found at runtime
cp $SRC_DIR/tools/tfrecord2idx $PREFIX/bin
