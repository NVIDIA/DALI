import tensorflow as tf

def DALIIterator():
    try:
        libndll_tf = 'libndll_tf.so'
        ndll_tf_module = tf.load_op_library(libndll_tf)
        ndll_tf = ndll_tf_module.ndll
    except Exception:
        print(libndll_tf + " not found: add /usr/local/lib/ to LD_LIBRARY_PATH")
        libndll_tf = '/opt/dali/build/ndll/libndll_tf.so'
        ndll_tf_module = tf.load_op_library(libndll_tf)
        ndll_tf = ndll_tf_module.ndll
    return ndll_tf
