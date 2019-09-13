#!/bin/bash

COMPILER=${CXX:-g++}
PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/ -I/home/awolant/Projects/tensorflow/ -I/home/awolant/.local/lib/python3.6/site-packages/tensorflow/include/ -I/home/awolant/Projects/DALI/include/ -I/home/awolant/Projects/DALI/"

DALI_CFLAGS=( $($PYTHON ./dali_compile_flags.py --cflags) )
DALI_LFLAGS=( $($PYTHON ./dali_compile_flags.py --lflags) )

TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


$COMPILER -std=c++11 -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} ${DALI_CFLAGS[@]} ${DALI_LFLAGS[@]}
