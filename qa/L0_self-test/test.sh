#!/bin/bash -ex

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

DALI_TEST_BENCHMARK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}")" && cd "../test_data/benchmark_images" && pwd )" \
DALI_TEST_IMAGES_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}")" && cd "../test_data/test_images" && pwd )" \
DALI_TEST_CAFFE_LMDB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}")" && cd "../test_data/test_db_images/train-lmdb-256x256" && pwd )" "$FULLPATH"
