#!/bin/bash -e
./test_mxnet.sh
./test_cupy.sh
./test_pytorch.sh
./test_pytorch_cupy.sh
