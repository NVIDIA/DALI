#!/bin/bash -e
bash -e ./test_nofw.sh
bash -e ./test_pytorch.sh
bash -e ./test_cupy.sh
bash -e ./test_jax.sh
