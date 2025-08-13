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

CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')
# make it on by default for CUDA 11.x
if [ "${CUDA_VERSION/./}" -ge 110 ]; then
  export WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT=ON
else
  export WITH_DYNAMIC_CUDA_TOOLKIT_DEFAULT=OFF
fi

# Create build directory for cmake and enter it
mkdir $SRC_DIR/build_core
cd $SRC_DIR/build_core
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
      -DBUILD_PYTHON=OFF                                  \
      -DPREBUILD_DALI_LIBS=OFF                            \
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
      -DNVIMGCODEC_DEFAULT_INSTALL_PATH=$PREFIX/opt/nvidia/nvimgcodec \
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
    "$PREFIX/lib/libavformat.so.61"
    "$PREFIX/lib/libavcodec.so.61"
    "$PREFIX/lib/libavfilter.so.10"
    "$PREFIX/lib/libavutil.so.59"
    "$PREFIX/lib/libswscale.so.8"
    "lib/libcvcuda.so.0"
    "lib/libnvcv_types.so.0"
)

DEPS_PATH=/usr/local/

if [ "$BUILD_NVCOMP" = "ON" ] && [ "$WITH_DYNAMIC_NVCOMP" = "ON" ]; then
    DEPS_LIST+=(
        "${DEPS_PATH}/cuda/lib64/libnvcomp.so.5"
    )
fi

PKGNAME_PATH=$PWD/dali/python/nvidia/dali
DEPS_LIB_DST_PATH=$PKGNAME_PATH/dali_deps_libs
mkdir -p $DEPS_LIB_DST_PATH

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
    patchedpath=$DEPS_LIB_DST_PATH/$patchedname
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
done < <(find $DEPS_LIB_DST_PATH -name '*.so*' -print0)
while IFS=  read -r -d $'\0'; do
    sofile_list+=("$REPLY")
done < <(find $PKGNAME_PATH -name '*.bin' -print0)
for ((i=0;i<${#sofile_list[@]};++i)); do
    sofile=${sofile_list[i]}
    patch_hashed_names $sofile &
done
wait
echo "Fixed hashed names"

# set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN$UPDIRS, $ORIGIN$UPDIRS/dali_deps_libs
find $PKGNAME_PATH -type f -name "*.so*" -o -name "*.bin" | while read FILE; do
    UPDIRS=$(dirname $(echo "$FILE" | sed "s|$PKGNAME_PATH||") | sed 's/[^\/][^\/]*/../g')
    echo "Setting rpath of $FILE to '\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/dali_deps_libs'"
    patchelf --set-rpath "\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/dali_deps_libs" $FILE
    patchelf --print-rpath $FILE
    if [[ "$FILE" == *".so"* ]]; then
        cp $FILE $PREFIX/lib/;
    fi
    if [[ "$FILE" == *"dali_deps_libs"* ]]; then
        mkdir -p $PREFIX/lib/dali_deps_libs/
        cp $FILE $PREFIX/lib/dali_deps_libs/;
    fi
    if [[ "$FILE" == *".bin"* ]]; then
        cp $FILE $PREFIX/bin/;
    fi
done

# copy generated headers for the bindings build
find -iname *.pb.h | while read FILE; do
   echo $FILE $PREFIX/include/$FILE
   mkdir -p $(dirname $PREFIX/include/$FILE)
   cp $FILE $PREFIX/include/$FILE
done
