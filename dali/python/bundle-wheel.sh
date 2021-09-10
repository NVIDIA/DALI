#!/bin/bash -e
#
# Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

INWHL=$(readlink -e $1)
STRIP_DEBUG=${2:-NO}
TEST_BUNDLED_LIBS=${3:-NO}
OUTWHLNAME=${4:-$(basename $INWHL)}
DEPS_PATH=${5:-/usr/local}
OUTDIR=/wheelhouse
SCRIPT_PATH=$(dirname $(readlink -f $0))

# For some reason the pip wheel builder inserts "-none-" into the tag even if you gave it an ABI name
OUTWHLNAME=${OUTWHLNAME//-none-/-}

PKGNAME=$(echo "$OUTWHLNAME" | sed 's/-.*$//')
PKGNAME_PATH=$(echo "$PKGNAME" | sed 's/_/\//' | sed 's/_.*$//')

if [[ -z "$INWHL" || ! -f "$INWHL" || -z "$PKGNAME" ]]; then
    echo "Usage: $0 <inputfile.whl>"
    exit 1
fi


#######################################################################
# ADD DEPENDENCIES INTO THE WHEEL
#
# auditwheel repair doesn't work correctly and is buggy
# so manually do the work of copying dependency libs and patchelfing
# and fixing RECORDS entries correctly
######################################################################

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    BASENAME=$(basename $1)
    INITNAME=$(echo $BASENAME | cut -f1 -d".")
    ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
    echo "$INITNAME-$HASH.$ENDNAME"
}

make_wheel_record() {
    FPATH=$1
    RECORD_FILE=$2
    TMPDIR=$3
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
    # if the RECORD file, then
    result="$FPATH,,"
    else
    HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
    FSIZE=$(ls -nl $FPATH | awk '{print $5}')
    result="$FPATH,sha256=$HASH,$FSIZE"
    fi
    flock $TMPDIR/dali_rec.lock echo $result>>$RECORD_FILE
}

DEPS_LIST=(
    "${DEPS_PATH}/lib64/libjpeg.so.62"
    "${DEPS_PATH}/lib/libjpeg.so.62"
    "${DEPS_PATH}/lib/libavformat.so.58"
    "${DEPS_PATH}/lib/libavcodec.so.58"
    "${DEPS_PATH}/lib/libavfilter.so.7"
    "${DEPS_PATH}/lib/libavutil.so.56"
    "${DEPS_PATH}/lib/libtiff.so.5"
    "${DEPS_PATH}/lib/libsndfile.so.1"
    "${DEPS_PATH}/lib/libFLAC.so.8"
    "${DEPS_PATH}/lib/libogg.so.0"
    "${DEPS_PATH}/lib/libvorbis.so.0"
    "${DEPS_PATH}/lib/libvorbisenc.so.2"
    "${DEPS_PATH}/lib/libopus.so.0"
    "${DEPS_PATH}/lib/libopenjp2.so.7"
    "${DEPS_PATH}/lib/libzstd.so.1"
    "${DEPS_PATH}/lib/libz.so.1"
)

TMPDIR=$(mktemp -d)
pushd $TMPDIR
unzip -q $INWHL
mkdir -p $PKGNAME_PATH/.libs
popd

strip_so () {
    local filepath=$1
    strip --strip-debug $filepath
}

if [[ "$STRIP_DEBUG" != "NO" ]]; then
    echo "Striping .so files from debug info"
    for f in $(find $TMPDIR -iname *.so); do
        strip_so $f &
    done
    wait
fi

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
    cp $filepath $TMPDIR/$patchedpath

    echo "Patching DT_SONAME field in $patchedpath"
    patchelf --set-soname $patchedname $TMPDIR/$patchedpath &
}

echo "Patching DT_SONAMEs..."
for filepath in "${DEPS_LIST[@]}"; do
    copy_and_patch $filepath
done
wait
echo "Patched DT_SONAMEs"

pushd $TMPDIR

patch_hashed_names() {
    local sofile=$1
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
                patchelf --replace-needed $origname $patchedname $sofile
            fi
        fi
    done
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

patch_rpath() {
    local FILE=$1
    UPDIRS=$(dirname $(echo "$FILE" | sed "s|$PKGNAME_PATH||") | sed 's/[^\/][^\/]*/../g')
    echo "Setting rpath of $FILE to '\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs'"
    patchelf --set-rpath "\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs" $FILE
    patchelf --print-rpath $FILE
}
echo "Fixing rpath of main files..."
# set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN$UPDIRS, $ORIGIN$UPDIRS/.libs
for ((i=0;i<${#sofile_list[@]};++i)); do
    sofile=${sofile_list[i]}
    patch_rpath $sofile &
done
wait
echo "Fixed rpath of main files"

patch_other_rpath() {
    local sofile=$1
    echo "Setting rpath of $sofile to " '$ORIGIN'
    patchelf --set-rpath '$ORIGIN' $sofile
    patchelf --print-rpath $sofile
}
echo "Fixing rpath of .lib files..."
# get list of files to iterate over
sofile_list=()
while IFS=  read -r -d $'\0'; do
    sofile_list+=("$REPLY")
done < <(find $PKGNAME_PATH/.libs -maxdepth 1 -type f -name "*.so*"  -print0)
# set RPATH of .libs/ files to $ORIGIN
for ((i=0;i<${#sofile_list[@]};++i)); do
    sofile=${sofile_list[i]}
    patch_other_rpath $sofile &
done
wait
echo "Fixed rpath of .lib files"

# correct the metadata in the dist-info/WHEEL, e.g.:
#Root-Is-Purelib: true
#Tag: cp27-cp27mu-none-manylinux1_x86_64
sed -i 's/\(Tag:.*\)-none-/\1-/;s/\(Root-Is-Purelib:\) true/\1 false/' ${PKGNAME}-*.dist-info/WHEEL

# regenerate the RECORD file with new hashes
RECORD_FILE=$(ls $PKGNAME-*.dist-info/RECORD)
echo "Generating new record file $RECORD_FILE"
rm -f $RECORD_FILE
# generate records for $PKGNAME_S folder
rec_list=()
while IFS=  read -r -d $'\0'; do
    rec_list+=("$REPLY")
done < <(find * -type f -print0)
for ((i=0;i<${#rec_list[@]};++i)); do
    FNAME=${rec_list[i]}
   make_wheel_record $FNAME $RECORD_FILE $TMPDIR &
done
wait
echo "$RECORD_FILE,," >> $RECORD_FILE
echo "Finished generating new record file $RECORD_FILE"

if [[ "$TEST_BUNDLED_LIBS" != "NO" ]]; then
    echo "Check bundled libs..."
    python ${SCRIPT_PATH}/../../tools/test_bundled_libs.py $(find ./ -iname *.so* | tr '\n' ' ')
fi

# zip up the new wheel into the wheelhouse
echo "Compressing wheel..."
mkdir -p $OUTDIR
rm -f $OUTDIR/$OUTWHLNAME
zip -rq $OUTDIR/$OUTWHLNAME *
echo "Finished compressing wheel"

# clean up
popd
rm -rf $TMPDIR
