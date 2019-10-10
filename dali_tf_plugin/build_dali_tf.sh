#!/bin/bash

COMPILER=${CXX:-g++}
PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/"

DALI_CFLAGS=( $($PYTHON ./dali_compile_flags.py --include_flags) )
DALI_LFLAGS=( $($PYTHON ./dali_compile_flags.py --lflags) )

TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


# Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
# https://github.com/tensorflow/tensorflow/issues/17316
# Do not remove it.
$COMPILER -std=c++11 -DNDEBUG -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} ${DALI_CFLAGS[@]} ${DALI_LFLAGS[@]}
