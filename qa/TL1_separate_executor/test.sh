#!/bin/bash -e
bash -e ./test_nofw.sh
bash -e ./test_tf.sh
bash -e ./test_pytorch.sh
bash -e ./test_mxnet.sh
bash -e ./test_paddle.sh