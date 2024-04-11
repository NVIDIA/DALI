#!/bin/bash -e
./test_cupy.sh
./test_pytorch.sh
./test_pytorch_cupy.sh
./test_jax.sh
