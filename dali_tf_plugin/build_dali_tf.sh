#!/bin/bash

COMPILER=${CXX:-g++}
PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/ -I../include"

DALI_STUB_SRC='./stub/dali_stub.cc'
DALI_STUB_LIB='libdali.so'

$COMPILER -std=c++11 -DNDEBUG -O2 -shared -fPIC ${DALI_STUB_SRC} -o ${DALI_STUB_LIB} ${INCL_DIRS}

TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
# https://github.com/tensorflow/tensorflow/issues/17316
# Do not remove it.
$COMPILER -std=c++11 -DNDEBUG -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${DALI_STUB_LIB} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}

rm ${DALI_STUB_LIB}
