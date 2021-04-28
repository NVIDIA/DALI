#!/bin/bash

COMPILER=${CXX:-g++}
PYTHON=${PYTHON:-python}
LIB_NAME=${1:-"libdali_tf_current.so"}
SRCS="daliop.cc dali_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/"

DALI_STUB_DIR=`mktemp -d -t "dali_stub_XXXXXX"`
DALI_STUB_SRC="${DALI_STUB_DIR}/dali_stub.cc"
DALI_STUB_LIB="${DALI_STUB_DIR}/libdali.so"
$PYTHON ../tools/stubgen.py ../include/dali/c_api.h --output "${DALI_STUB_SRC}"

DALI_CFLAGS="-I../include -I.."
DALI_LFLAGS="-L${DALI_STUB_DIR} -ldali"

$COMPILER -std=c++11 -DNDEBUG -O2 -shared -fPIC ${DALI_STUB_SRC} -o ${DALI_STUB_LIB} ${INCL_DIRS} ${DALI_CFLAGS}

TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
# https://github.com/tensorflow/tensorflow/issues/17316
# Do not remove it.
$COMPILER -std=c++11 -DNDEBUG -O2 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${DALI_CFLAGS} ${DALI_LFLAGS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}

# Cleaning up stub dir
rm -rf "${DALI_STUB_DIR}"
