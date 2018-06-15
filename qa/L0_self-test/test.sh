#!/bin/bash -e

pushd ../..

cd build-*$PYV*
NDLL_TEST_CAFFE_LMDB_PATH="/data/imagenet/train-lmdb-256x256" ./ndll/run_tests

popd
