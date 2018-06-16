try:
    from dali.backend_impl.tfrecord import *
except ImportError:
    raise RuntimeError('DALI was not compiled with TFRecord support.'
            ' Use BUILD_PROTOBUF=ON CMake option to enable TFRecord support')
