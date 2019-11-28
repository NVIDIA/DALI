# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
from tensorflow.python.client import device_lib
import nvidia.dali.plugin.tf as dali_tf
from nose import SkipTest


def skip_for_incompatible_tf():
    if not dali_tf.dataset_compatible_tensorflow():
        raise SkipTest('This feature is enabled for TF 1.15 and higher')


def num_available_gpus():
    local_devices = device_lib.list_local_devices()
    num_gpus = sum(1 for device in local_devices if device.device_type == 'GPU')
    if num_gpus not in [1, 2, 4, 8]:
        raise RuntimeError('Unsupported number of GPUs. This test can run on: 1, 2, 4, 8 GPUs.')
    return num_gpus


def available_gpus():
    devices = []
    for device_id in range(num_available_gpus()):
        devices.append('/gpu:{0}'.format(device_id))
    return devices


def dataset_options():
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False   
    except:
        print('Could not set TF Dataset Options')

    return options