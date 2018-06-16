import tensorflow as tf

def DALIIterator():
    try:
        libdali_tf = 'libdali_tf.so'
        dali_tf_module = tf.load_op_library(libdali_tf)
        dali_tf = dali_tf_module.dali
    except Exception:
        print(libdali_tf + " not found: add /usr/local/lib/ to LD_LIBRARY_PATH")
        libdali_tf = '/opt/dali/build/dali/libdali_tf.so'
        dali_tf_module = tf.load_op_library(libdali_tf)
        dali_tf = dali_tf_module.dali
    return dali_tf
