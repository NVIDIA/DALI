#!/bin/bash -e
bash -e ./test_tf.sh
bash -e ./test_paddle.sh
bash -e ./test_pytorch.sh
bash -e ./test_jax.sh
