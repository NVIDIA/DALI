#!/bin/bash -ex

source ../setup_test.sh

BINNAME=dali_test.bin

for DIRNAME in \
  "../../build/dali/python/nvidia/dali" \
  "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
do
    if [ -x "$DIRNAME/test/$BINNAME" ]; then
        FULLPATH="$DIRNAME/test/$BINNAME"
        break
    fi
done

if [[ -z "$FULLPATH" ]]; then
    echo "ERROR: $BINNAME not found"
    exit 1
fi

DALI_TEST_CAFFE_LMDB_PATH="/data/imagenet/train-lmdb-256x256" "$FULLPATH"
