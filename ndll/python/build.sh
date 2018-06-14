#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#

# Portions of this file are derived from https://github.com/pytorch/builder/blob/d5e62b676b5d3b6c5dba35a4b5ac227bd6d3563b/manywheel/build.sh
#
# Copyright (c) 2016, Hugh Perkins
# Copyright (c) 2016, Soumith Chintala
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#######################################################################
# ADD DEPENDENCIES INTO THE WHEEL
#
# auditwheel repair doesn't work correctly and is buggy
# so manually do the work of copying dependency libs and patchelfing
# and fixing RECORDS entries correctly
######################################################################

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
    echo $1
    else
    INITNAME=$(echo $BASENAME | cut -f1 -d".")
    ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
    echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

make_wheel_record() {
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
    # if the RECORD file, then
    echo "$FPATH,,"
    else
    HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
    FSIZE=$(ls -nl $FPATH | awk '{print $5}')
    echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

if [[ $CUDA_VERSION == "8.0" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.8.0.61"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.8.0.61"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "/usr/lib64/libgomp.so.1"
)

DEPS_SONAME=(
    "libcudart.so.8.0"
    "libnvToolsExt.so.1"
    "libnvrtc.so.8.0"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)

elif [[ $CUDA_VERSION == "9.0" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.9.0"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.9.0"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "/usr/lib64/libgomp.so.1"
)

DEPS_SONAME=(
    "libcudart.so.9.0"
    "libnvToolsExt.so.1"
    "libnvrtc.so.9.0"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
elif [[ $CUDA_VERSION == "9.1" ]]; then
DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.9.1"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.9.1"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "/usr/lib64/libgomp.so.1"
)

DEPS_SONAME=(
    "libcudart.so.9.1"
    "libnvToolsExt.so.1"
    "libnvrtc.so.9.1"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
else
    echo "Unknown cuda version $CUDA_VERSION"
    exit 1
fi

mkdir -p /$WHEELHOUSE_DIR
cp $PYTORCH_DIR/$WHEELHOUSE_DIR/*.whl /$WHEELHOUSE_DIR
mkdir /tmp_dir
pushd /tmp_dir

for whl in /$WHEELHOUSE_DIR/torch*linux*.whl; do
    rm -rf tmp
    mkdir -p tmp
    cd tmp
    cp $whl .

    unzip -q $(basename $whl)
    rm -f $(basename $whl)

    # copy over needed dependent .so files over and tag them with their hash
    patched=()
    for filepath in "${DEPS_LIST[@]}"
    do
    filename=$(basename $filepath)
    destpath=torch/lib/$filename
    if [[ "$filepath" != "$destpath" ]]; then
        cp $filepath $destpath
    fi

    patchedpath=$(fname_with_sha256 $destpath)
    patchedname=$(basename $patchedpath)
    if [[ "$destpath" != "$patchedpath" ]]; then
        mv $destpath $patchedpath
    fi
    patched+=("$patchedname")
    echo "Copied $filepath to $patchedpath"
    done

    echo "patching to fix the so names to the hashed names"
    for ((i=0;i<${#DEPS_LIST[@]};++i));
    do
    find torch -name '*.so*' | while read sofile; do
        origname=${DEPS_SONAME[i]}
        patchedname=${patched[i]}
        if [[ "$origname" != "$patchedname" ]]; then
        set +e
        patchelf --print-needed $sofile | grep $origname 2>&1 >/dev/null
        ERRCODE=$?
        set -e
        if [ "$ERRCODE" -eq "0" ]; then
            echo "patching $sofile entry $origname to $patchedname"
            patchelf --replace-needed $origname $patchedname $sofile
        fi
        fi
    done
    done

    # set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
    find torch -maxdepth 1 -type f -name "*.so*" | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/lib'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/lib' $sofile
    patchelf --print-rpath $sofile
    done

    # set RPATH of lib/ files to $ORIGIN
    find torch/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN'
    patchelf --set-rpath '$ORIGIN' $sofile
    patchelf --print-rpath $sofile
    done


    # regenerate the RECORD file with new hashes 
    record_file=`echo $(basename $whl) | sed -e 's/-cp.*$/.dist-info\/RECORD/g'`
    echo "Generating new record file $record_file"
    rm -f $record_file
    # generate records for torch folder
    find torch -type f | while read fname; do
    echo $(make_wheel_record $fname) >>$record_file
    done
    # generate records for torch-[version]-dist-info folder
    find torch*dist-info -type f | while read fname; do
    echo $(make_wheel_record $fname) >>$record_file
    done

    # zip up the wheel back
    zip -rq $(basename $whl) torch*

    # replace original wheel
    rm -f $whl
    mv $(basename $whl) $whl
    cd ..
    rm -rf tmp
done

mkdir -p /remote/$WHEELHOUSE_DIR
cp /$WHEELHOUSE_DIR/torch*.whl /remote/$WHEELHOUSE_DIR/

