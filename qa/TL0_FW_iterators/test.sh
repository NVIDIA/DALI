#!/bin/bash -e

# Tensorflow tests are incompatible with Python 3.13.
# Check Python version and run test accordingly.
set +e
python -c '
import sys
if sys.version_info == (3, 13):
    sys.exit(1)
'
if [ $? -eq 0 ]; then
    bash -e ./test_tf.sh
fi
set -e

bash -e ./test_paddle.sh
bash -e ./test_pytorch.sh
bash -e ./test_jax.sh
