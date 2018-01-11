try:
    from ndll.ndll_backend.tfrecord import *
except ImportError:
    raise RuntimeError('NDLL was not compiled with TFRecord support.'
            ' Use BUILD_PROTOBUF=ON CMake option to enable TFRecord support')
