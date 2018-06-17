#!/bin/bash -e
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

INWHL=$(readlink -e $1)
OUTDIR=/wheelhouse

OUTWHLNAME=$(basename $INWHL)
# For some reason the pip wheel builder inserts "-none-" into the tag even if you gave it an ABI name
OUTWHLNAME=${OUTWHLNAME//-none-/-}

PKGNAME=$(echo "$OUTWHLNAME" | sed 's/-.*$//')
PKGNAME_PATH=$(echo "$PKGNAME" | sed 's/_/\//' )

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
    "/usr/local/lib/liblmdb.so"
    "dali/libdali.so"
    "/usr/local/lib/libnvjpeg.so.9.0"
    "/usr/local/lib/libopencv_core.so.3.1"
    "/usr/local/lib/libopencv_imgcodecs.so.3.1"
    "/usr/local/lib/libopencv_imgproc.so.3.1"
    "/usr/local/lib/libprotobuf.so.15"
    "/usr/local/lib/libturbojpeg.so.0"
)

DEPS_SONAME=(
    "liblmdb.so"
    "libdali.so"
    "libnvjpeg.so.9.0"
    "libopencv_core.so.3.1"
    "libopencv_imgcodecs.so.3.1"
    "libopencv_imgproc.so.3.1"
    "libprotobuf.so.15"
    "libturbojpeg.so.0"
)

TMPDIR=$(mktemp -d)
pushd $TMPDIR
unzip -q $INWHL
mkdir -p $PKGNAME_PATH/.libs
popd

# copy over needed dependent .so files over and tag them with their hash
patched=()
for filepath in "${DEPS_LIST[@]}"; do
    filename=$(basename $filepath)
    patchedname=$(fname_with_sha256 $filepath)
    patchedpath=$PKGNAME_PATH/.libs/$patchedname
    cp $filepath $TMPDIR/$patchedpath
    patched+=("$patchedname")
    echo "Copied $filepath to $patchedpath"
done

pushd $TMPDIR

echo "patching to fix the so names to the hashed names"
find $PKGNAME_PATH -name '*.so*' -o -name '*.bin' | while read sofile; do
    for ((i=0;i<${#DEPS_LIST[@]};++i)); do
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

# set RPATH of backend_impl.so and similar to $ORIGIN, $ORIGIN/.libs
find $PKGNAME_PATH -maxdepth 1 -type f -name "*.so*" | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/.libs'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/.libs' $sofile
    patchelf --print-rpath $sofile
done

# set RPATH of test/*.bin to $ORIGIN, $ORIGIN/.libs
find $PKGNAME_PATH/test -maxdepth 1 -type f -name "*.bin" | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/.libs'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../.libs' $sofile
    patchelf --print-rpath $sofile
done

# set RPATH of .libs/ files to $ORIGIN
find $PKGNAME_PATH/.libs -maxdepth 1 -type f -name "*.so*" | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN'
    patchelf --set-rpath '$ORIGIN' $sofile
    patchelf --print-rpath $sofile
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

