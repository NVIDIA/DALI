import tensorflow as tf

def ndllTFOp():
    try:
        libndllop = 'libndllop.so'
        ndllop_module = tf.load_op_library(libndllop)
        ndllop = ndllop_module.ndll
    except Exception:
        print(libndllop + " not install: add /usr/local/lib to LD_LIBRARY_PATH")
        libndllop = '/opt/ndll/build/ndll/libndllop.so'
        ndllop_module = tf.load_op_library(libndllop)
        ndllop = ndllop_module.ndll
    return ndllop
