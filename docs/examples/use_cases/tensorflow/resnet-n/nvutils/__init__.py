#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .runner import train
from .runner_ctl import train_ctl
from .runner import predict
from .runner_ctl import predict_ctl
from .cmdline import parse_cmdline
import os, sys, random
import tensorflow as tf
import horovod.tensorflow.keras as hvd

def init():
  gpu_thread_count = 2
  os.environ['TF_GPU_THREAD_MODE']  = 'gpu_private'
  os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
  os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  hvd.init()
  if hvd.rank() == 0:
    print('PY', sys.version)
    print('TF', tf.version.VERSION)

