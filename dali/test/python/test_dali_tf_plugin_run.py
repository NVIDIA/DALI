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

import numpy as np
from subprocess import call
import os.path
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
from test_utils import get_dali_extra_path
from nose.tools import raises

try:
    tf.compat.v1.disable_eager_execution()
except:
    pass

test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

IMG_SIZE = 227
NUM_GPUS = 1

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.decoders.Image(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                           output_dtype = types.FLOAT,
                                           crop = (227, 227),
                                           mean = [128., 128., 128.],
                                           std = [1., 1., 1.])
        self.uniform = ops.random.Uniform(range = (0.0, 1.0))
        self.resize_rng = ops.random.Uniform(range = (256, 480))

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.resize_rng())
        output = self.cmn(images,
                          crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels.gpu())

class CaffeReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(CaffeReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.readers.Caffe(path = lmdb_folder,
                                       random_shuffle = True, shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input()
        return self.base_define_graph(images, labels)

def get_batch_dali(batch_size, pipe_type, label_type, num_gpus=1):
    pipes = [pipe_type(batch_size=batch_size, num_threads=2, device_id = device_id, num_gpus = num_gpus) for device_id in range(num_gpus)]

    daliop = dali_tf.DALIIterator()
    images = []
    labels = []
    for d in range(NUM_GPUS):
        with tf.device('/gpu:%i' % d):
            image, label = daliop(pipeline = pipes[d],
                shapes = [(batch_size, 3, 227, 227), ()],
                dtypes = [tf.int32, label_type],
                device_id = d)
            images.append(image)
            labels.append(label)

    return [images, labels]

def test_dali_tf_op(pipe_type=CaffeReadPipeline, batch_size=16, iterations=32):
    test_batch = get_batch_dali(batch_size, pipe_type, tf.int32)
    try:
        from tensorflow.compat.v1 import GPUOptions
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import Session
    except:
        # Older TF versions don't have compat.v1 layer
        from tensorflow import GPUOptions
        from tensorflow import ConfigProto
        from tensorflow import Session

    gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = ConfigProto(gpu_options=gpu_options)
    with Session(config=config) as sess:
        for i in range(iterations):
            imgs, labels = sess.run(test_batch)
            # Testing correctness of labels
            for label in labels:
                ## labels need to be integers
                assert(np.equal(np.mod(label, 1), 0).all())
                assert((label >= 0).all())
                assert((label <= 999).all())


class PythonOperatorPipeline(Pipeline):
    def __init__(self):
        super(PythonOperatorPipeline, self).__init__(1, 1, 0, 0)
        self.python_op = ops.PythonFunction(function=lambda: np.zeros((3, 3, 3)))

    def define_graph(self):
        return self.python_op()


@raises(RuntimeError)
def test_python_operator_error():
    daliop = dali_tf.DALIIterator()
    pipe = PythonOperatorPipeline()
    with tf.device('/cpu:0'):
        output = daliop(pipeline=pipe, shapes=[(1, 3, 3, 3)], dtypes=[tf.float32], device_id=0)
