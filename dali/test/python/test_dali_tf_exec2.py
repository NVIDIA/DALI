# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os.path
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
from nose_utils import with_setup
from test_utils_tensorflow import skip_inputs_for_incompatible_tf
from test_utils import get_dali_extra_path


test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, "db", "lmdb")


@pipeline_def(
    enable_conditionals=True,
    batch_size=5,
    num_threads=4,
    device_id=0,
    exec_dynamic=True,
)
def dali_exec2_pipeline():
    iter_id = fn.external_source(source=lambda x: np.array(x.iteration), batch=False)
    if iter_id & 1 == 0:
        output = types.Constant(np.array(-1), device="gpu")
    else:
        output = types.Constant(np.array(1), device="gpu")
    return output.cpu()


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_exec2():
    """Test that exec_dynamic is propagated to DALI pipeline from dali_tf.DALIDatasetWithInputs"""
    # From Tensorflow's perspective, this is a CPU pipeline
    with tf.device("/cpu:0"):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=dali_exec2_pipeline(),
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


@pipeline_def(num_threads=4, exec_dynamic=True)
def daliop_pipe():
    jpegs, labels = fn.readers.caffe(path=lmdb_folder, random_shuffle=False)
    imgs = fn.decoders.image(jpegs, device="mixed")
    imgs = fn.resize(imgs, size=(100, 100))
    shape = imgs.shape(dtype=types.UINT32)
    return imgs.cpu(), shape


def get_batch_dali(batch_size):
    pipe = daliop_pipe(batch_size=batch_size, num_threads=4, device_id=0)

    daliop = dali_tf.DALIIterator()
    images = []
    labels = []
    with tf.device("/cpu:0"):
        image, label = daliop(
            pipeline=pipe,
            shapes=[
                (batch_size, 100, 100, 3),
                (
                    batch_size,
                    3,
                ),
            ],
            dtypes=[tf.uint8, tf.int32],
            device_id=0,
        )
        images.append(image)
        labels.append(label)

    return [images, labels]


def test_tf_op():
    """Test that exec_dynamic is propagated to DALI pipeline from dali_tf.DALIIterator"""
    try:
        tf.compat.v1.disable_eager_execution()
    except ModuleNotFoundError:
        pass

    batch_size = 8
    iterations = 2
    test_batch = get_batch_dali(batch_size)
    try:
        from tensorflow.compat.v1 import Session
    except ImportError:
        # Older TF versions don't have compat.v1 layer
        from tensorflow import Session

    with Session() as sess:
        for i in range(iterations):
            imgs, shapes = sess.run(test_batch)
            for img, shape in zip(imgs, shapes):
                for i in range(batch_size):
                    assert tuple(img[i].shape) == tuple(shape[i])
