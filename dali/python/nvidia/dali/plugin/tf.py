import tensorflow as tf
import os

def DALIIterator():
    libdali_tf = os.path.dirname(os.path.realpath(__file__)) + '/libdali_tf.so'
    dali_tf_module = tf.load_op_library(libdali_tf)
    dali_tf = dali_tf_module.dali
    return dali_tf
