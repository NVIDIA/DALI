# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.pipeline.experimental import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from nose_utils import with_setup
from test_utils_tensorflow import skip_inputs_for_incompatible_tf


@with_setup(skip_inputs_for_incompatible_tf)
def test_both_tf_and_dali_conditionals():
    @pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0)
    def dali_conditional_pipeline():
        iter_id = fn.external_source(source=lambda x: np.array(x.iteration), batch=False)
        if iter_id & 1 == 0:
            output = types.Constant(np.array(-1), device="cpu")
        else:
            output = types.Constant(np.array(1), device="cpu")
        return output

    with tf.device("/cpu:0"):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=dali_conditional_pipeline(),
            batch_size=5,
            output_shapes=(5,),
            output_dtypes=(tf.int32),
            num_threads=4,
            device_id=0,
        )

        @tf.function
        def tf_function_with_conditionals(dali_dataset):
            negative = tf.constant(0)
            positive = tf.constant(0)
            for input in dali_dataset:
                if tf.reduce_sum(input) < 0:
                    negative = negative + 1
                else:
                    positive = positive + 1
            return negative, positive

        pos, neg = tf_function_with_conditionals(dali_dataset.take(5))
        assert pos == 3
        assert neg == 2
