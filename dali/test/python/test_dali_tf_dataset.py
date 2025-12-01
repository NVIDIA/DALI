# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf
from nose_utils import raises

from test_utils_tensorflow import get_image_pipeline


@raises(ValueError, "Two structures don't have the same sequence length*length 3*length 2")
def test_different_num_shapes_dtypes():
    batch_size = 12
    num_threads = 4

    dataset_pipe, shapes, dtypes = get_image_pipeline(batch_size, num_threads, "cpu")
    dtypes = tuple(dtypes[0:2])

    with tf.device("/cpu:0"):
        dali_tf.DALIDataset(
            pipeline=dataset_pipe,
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            num_threads=num_threads,
        )


@raises(RuntimeError, "some operators*cannot be used with TensorFlow Dataset API and DALIIterator")
def test_python_operator_not_allowed_in_tf_dataset_error():
    pipeline = Pipeline(1, 1, 0, exec_pipelined=False, exec_async=False)
    with pipeline:
        output = fn.python_function(function=lambda: np.zeros((3, 3, 3)))
        pipeline.set_outputs(output)

    shapes = (1, 3, 3, 3)
    dtypes = tf.float32

    with tf.device("/cpu:0"):
        _ = dali_tf.DALIDataset(
            pipeline=pipeline,
            batch_size=1,
            output_shapes=shapes,
            output_dtypes=dtypes,
            num_threads=1,
            device_id=0,
        )
