#!/bin/bash -e
./test_nofw.sh
./test_tf.sh
./test_pytorch.sh
./test_mxnet.sh
./test_paddle.sh