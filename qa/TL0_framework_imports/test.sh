#!/bin/bash -e
bash -e ./test_tf.sh
bash -e ./test_mxnet.sh
bash -e ./test_pytorch.sh
bash -e ./test_no_fw.sh
