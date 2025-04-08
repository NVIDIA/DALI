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

import numpy as np
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import random
import tensorflow as tf
from nvidia.dali.pipeline import pipeline_def

try:
    from tensorflow.compat.v1 import Session
except Exception:
    # Older TF versions don't have compat.v1 layer
    from tensorflow import Session


@pipeline_def()
def get_dali_pipe(value):
    data = types.Constant(value)
    return data


def get_data(batch_size, value):
    pipe = get_dali_pipe(batch_size=batch_size, device_id=None, num_threads=1, value=value)
    daliop = dali_tf.DALIIterator()
    out = []
    with tf.device("/cpu"):
        data = daliop(
            pipeline=pipe,
            shapes=[batch_size],
            dtypes=[tf.int32],
            device_id=None,
        )
        out.append(data)
    return [out]


def test_dali_tf_op_cpu_only():
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    value = random.randint(0, 1000)
    batch_size = 3
    test_batch = get_data(batch_size, value)
    with Session() as sess:
        data = sess.run(test_batch)
        assert (data == np.array([value] * batch_size)).all()
