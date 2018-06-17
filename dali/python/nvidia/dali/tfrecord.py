# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
try:
    from nvidia.dali.backend_impl.tfrecord import *
except ImportError:
    raise RuntimeError('DALI was not compiled with TFRecord support.'
            ' Use BUILD_PROTOBUF=ON CMake option to enable TFRecord support')
