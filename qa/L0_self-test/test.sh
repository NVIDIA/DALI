#!/bin/bash -e

export NDLL_TEST_CAFFE_LMDB_PATH="/data/imagenet/train-lmdb-256x256"

cd /opt/dali/build-*$PYV*
./ndll/run_tests
