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
      -DBUILD_NVTX=${BUILD_NVTX}                          \
      -DBUILD_PYTHON=${BUILD_PYTHON:-ON}                  \
      -DPYTHON_STUBGEN_INTERPRETER=${PYTHON}              \
      -DBUILD_LMDB=${BUILD_LMDB:-ON}                      \
      -DBUILD_JPEG_TURBO=${BUILD_JPEG_TURBO:-ON}          \
      -DBUILD_OPENCV=${BUILD_OPENCV:-ON}                  \
      -DBUILD_PROTOBUF=${BUILD_PROTOBUF:-ON}              \
      -DBUILD_NVJPEG=${BUILD_NVJPEG:-ON}                  \
      -DBUILD_NVJPEG2K=${BUILD_NVJPEG2K}                  \
      -DBUILD_LIBTIFF=${BUILD_LIBTIFF:-ON}                \
      -DBUILD_LIBSND=${BUILD_LIBSND:-ON}                  \
      -DBUILD_LIBTAR=${BUILD_LIBTAR:-ON}                  \
      -DBUILD_FFTS=${BUILD_FFTS:-ON}                      \
      -DBUILD_NVOF=${BUILD_NVOF:-ON}                      \
      -DBUILD_NVDEC=${BUILD_NVDEC:-ON}                    \
      -DBUILD_NVML=${BUILD_NVML:-ON}                      \
      -DBUILD_CUFILE=${BUILD_CUFILE:-ON}                  \
      -DBUILD_NVCOMP=${BUILD_NVCOMP}                      \
      -DBUILD_CVCUDA=${BUILD_CVCUDA:-ON}                  \
      -DLINK_LIBCUDA=${LINK_LIBCUDA:-OFF}                 \
      -DWITH_DYNAMIC_CUDA_TOOLKIT=${WITH_DYNAMIC_CUDA_TOOLKIT:-${WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT}}\
      -DWITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG:-ON}     \
      -DWITH_DYNAMIC_CUFFT=${WITH_DYNAMIC_CUFFT:-ON}       \
      -DWITH_DYNAMIC_NPP=${WITH_DYNAMIC_NPP:-ON}           \
      -DVERBOSE_LOGS=${VERBOSE_LOGS:-OFF}                 \
      -DWERROR=${WERROR:-ON}                              \
      -DBUILD_WITH_ASAN=${BUILD_WITH_ASAN:-OFF}           \
      -DBUILD_WITH_LSAN=${BUILD_WITH_LSAN:-OFF}           \
      -DBUILD_WITH_UBSAN=${BUILD_WITH_UBSAN:-OFF}         \
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
    "$PREFIX/lib/libavformat.so.59"
    "$PREFIX/lib/libavcodec.so.59"
    "$PREFIX/lib/libavfilter.so.8"
    "$PREFIX/lib/libavutil.so.57"
    "$PREFIX/lib/libswscale.so.6"
    "lib/libcvcuda.so.0"
    "lib/libnvcv_types.so.0"
)

DEPS_SONAME=(
    "libavformat.so.59"
    "libavcodec.so.59"
    "libavfilter.so.8"
    "libavutil.so.57"
    "libswscale.so.6"
    "libcvcuda.so.0"
    "libnvcv_types.so.0"
)

if [ "$BUILD_NVCOMP" = "ON" ]; then
    DEPS_LIST+=(
        "${DEPS_PATH}/cuda/lib64/libnvcomp.so"
        "${DEPS_PATH}/cuda/lib64/libnvcomp_gdeflate.so"
        "${DEPS_PATH}/cuda/lib64/libnvcomp_bitcomp.so"
    )

    DEPS_SONAME+=(
        "libnvcomp.so"
        "libnvcomp_gdeflate.so"
        "libnvcomp_bitcomp.so"
    )
fi

PKGNAME_PATH=dali/python/nvidia/dali/
mkdir -p $PKGNAME_PATH/.libs

# copy needed dependent .so files and tag them with their hash
original=()
patched=()

copy_and_patch() {
    local filepath=$1
    filename=$(basename $filepath)

    if [[ ! -f "$filepath" ]]; then
        echo "Didn't find $filename, skipping..."
        return
    fi
    patchedname=$(fname_with_sha256 $filepath)
    patchedpath=$PKGNAME_PATH/.libs/$patchedname
    original+=("$filename")
    patched+=("$patchedname")

    echo "Copying $filepath to $patchedpath"
    cp $filepath $patchedpath

    echo "Patching DT_SONAME field in $patchedpath"
    patchelf --set-soname $patchedname $patchedpath &
}

echo "Patching DT_SONAMEs..."
for filepath in "${DEPS_LIST[@]}"; do
    copy_and_patch $filepath
done
wait
echo "Patched DT_SONAMEs"

patch_hashed_names() {
    local sofile=$1
    local patch_cmd=""
    needed_so_files=$(patchelf --print-needed $sofile)
    for ((j=0;j<${#original[@]};++j)); do
        origname=${original[j]}
        patchedname=${patched[j]}
        if [[ "$origname" != "$patchedname" ]]; then
            set +e
            echo $needed_so_files | grep $origname 2>&1 >/dev/null
            ERRCODE=$?
            set -e
            if [ "$ERRCODE" -eq "0" ]; then
                echo "patching $sofile entry $origname to $patchedname"
                patch_cmd="$patch_cmd --replace-needed $origname $patchedname"
            fi
        fi
    done
    if [ -n "$patch_cmd" ]; then
        echo "running $patch_cmd on $sofile"
        patchelf $patch_cmd $sofile
    fi
}
echo "Patching to fix the so names to the hashed names..."
# get list of files to iterate over
sofile_list=()
while IFS=  read -r -d $'\0'; do
    sofile_list+=("$REPLY")
done < <(find $PKGNAME_PATH -name '*.so*' -print0)
while IFS=  read -r -d $'\0'; do
    sofile_list+=("$REPLY")
done < <(find $PKGNAME_PATH -name '*.bin' -print0)
for ((i=0;i<${#sofile_list[@]};++i)); do
    sofile=${sofile_list[i]}
    patch_hashed_names $sofile &
done
wait
echo "Fixed hashed names"

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

# Move tfrecord2idx to host env so it can be found at runtime
cp $SRC_DIR/tools/tfrecord2idx $PREFIX/bin
