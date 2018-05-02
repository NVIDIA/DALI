#!/bin/bash
set -e
# cd "$( dirname "${BASH_SOURCE[0]}" )"

export NDLL_TEST_CAFFE_LMDB_PATH="/data/imagenet/train-lmdb-256x256"

cd /opt/ndll/build
./ndll/run_tests
