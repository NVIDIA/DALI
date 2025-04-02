# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nose_utils  # noqa:F401  - for Python 3.10
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from test_utils_tensorflow import skip_for_incompatible_tf
import tensorflow as tf
import random
import numpy as np


@pipeline_def()
def get_dali_pipe(value):
    data = types.Constant(value)
    return data


def test_dali_tf_dataset_cpu_only():
    skip_for_incompatible_tf()
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        pass

    batch_size = 3
    value = random.randint(0, 1000)
    pipe = get_dali_pipe(batch_size=batch_size, device_id=None, num_threads=1, value=value)
    with tf.device("/cpu"):
        ds = dali_tf.DALIDataset(
            pipe,
            device_id=None,
            batch_size=1,
            output_dtypes=tf.int32,
            output_shapes=[1],
        )
    ds = iter(ds)
    data = next(ds)
    assert data == np.array([value])
