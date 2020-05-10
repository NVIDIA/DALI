#!/bin/bash -e
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
OUTDIR=/wheelhouse

OUTWHLNAME=$(basename $INWHL)
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
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
    # if the RECORD file, then
    echo "$FPATH,,"
    else
    HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
    FSIZE=$(ls -nl $FPATH | awk '{print $5}')
    echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

DEPS_LIST=(
    "/usr/local/lib64/libjpeg.so.62"
    "/usr/local/lib/libavformat.so.58"
    "/usr/local/lib/libavcodec.so.58"
    "/usr/local/lib/libavfilter.so.7"
    "/usr/local/lib/libavutil.so.56"
    "/usr/local/lib/libtiff.so.5"
    "/usr/local/lib/libsndfile.so.1"
    "/usr/local/lib/libFLAC.so.8"
    "/usr/local/lib/libogg.so.0"
    "/usr/local/lib/libvorbis.so.0"
    "/usr/local/lib/libvorbisenc.so.2"
)

DEPS_SONAME=(
    "libjpeg.so.62"
    "libavformat.so.58"
    "libavcodec.so.58"
    "libavfilter.so.7"
    "libavutil.so.56"
    "libtiff.so.5"
    "libsndfile.so.1"
    "libFLAC.so.8"
    "libogg.so.0"
    "libvorbis.so.0"
    "libvorbisenc.so.2"
)

TMPDIR=$(mktemp -d)
pushd $TMPDIR
unzip -q $INWHL
mkdir -p $PKGNAME_PATH/.libs
popd

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
patch_elf="python $SCRIPTPATH/patcher.py"
echo ${patch_elf}
# copy over needed dependent .so files over and tag them with their hash
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
    cp $filepath $TMPDIR/$patchedpath
done

pushd $TMPDIR

echo "patching to fix the so names to the hashed names, patching RPATH/RUNPATH"
find $PKGNAME_PATH -name '*.so*' -o -name '*.bin' | while read sofile; do
    new_so_path=''
    if [ -n ${sofile%%*\.lib*} ]; then
        # set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN$UPDIRS, $ORIGIN$UPDIRS/.libs
        UPDIRS=$(dirname $(echo "$sofile" | sed "s|$PKGNAME_PATH||") | sed 's/[^\/][^\/]*/../g')
        new_so_path="\$ORIGIN:\$ORIGIN$UPDIRS:\$ORIGIN$UPDIRS/.libs"
    else
        # set RPATH of .libs/ files to $ORIGIN
        new_so_path="$ORIGIN"
    fi

    echo "*************************"
    echo "patching $sofile:"
    echo "from:             ${DEPS_SONAME[@]//[$'\t\r\n']}"
    echo "to:               ${patched[@]//[$'\t\r\n']}"
    echo "Setting rpath to: $new_so_path"
    if [[ ${patched[@]} =~ $(basename $sofile) ]]; then
        patchedname=$(basename $sofile)
        echo "DT_SONAME to:     $patchedname"
        ${patch_elf} --replace-needed ${DEPS_SONAME[@]} ${patched[@]} --set-rpath $new_so_path --set-soname $patchedname --file $sofile
    else
        ${patch_elf} --replace-needed ${DEPS_SONAME[@]} ${patched[@]} --set-rpath $new_so_path --file $sofile
    fi
    ${patch_elf} --print-rpath --file $sofile
    echo "*************************"
done

# correct the metadata in the dist-info/WHEEL, e.g.:
#Root-Is-Purelib: true
#Tag: cp27-cp27mu-none-manylinux1_x86_64
sed -i 's/\(Tag:.*\)-none-/\1-/;s/\(Root-Is-Purelib:\) true/\1 false/' ${PKGNAME}-*.dist-info/WHEEL

# regenerate the RECORD file with new hashes
RECORD_FILE=$(ls $PKGNAME-*.dist-info/RECORD)
echo "Generating new record file $RECORD_FILE"
rm -f $RECORD_FILE
# generate records for $PKGNAME_S folder
find * -type f | while read FNAME; do
    echo $(make_wheel_record $FNAME) >>$RECORD_FILE
done
echo "$RECORD_FILE,," >> $RECORD_FILE

# zip up the new wheel into the wheelhouse
mkdir -p $OUTDIR
rm -f $OUTDIR/$OUTWHLNAME
zip -rq $OUTDIR/$OUTWHLNAME *

# clean up
popd
rm -rf $TMPDIR
