# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from test_utils import check_batch

import numpy as np

num_classes = 20
batch_size = 10


def insert_as_axis(target, value, axis, total_axes):
    return target[0:axis] + (value,) + target[axis:total_axes]

class OneHotPipeline(Pipeline):
    def __init__(self, num_classes, input, axis=-1, num_threads=1):
        super(OneHotPipeline, self).__init__(batch_size,
                                             num_threads,
                                             0)
        sample_dim = len(input[0].shape)
        self.ext_src = ops.ExternalSource(source=[input], cycle=True, layout="ABCD"[0:sample_dim])
        self.one_hot = ops.OneHot(num_classes=num_classes, axis=axis, dtype=types.INT32, device="cpu")

    def define_graph(self):
        self.data = self.ext_src()
        return self.one_hot(self.data)

def one_hot_3_axes(input, axis):
    total_axes = len(input[0].shape)
    assert total_axes == 3
    axis = axis if axis >= 0 else total_axes
    shapes = []
    results = []
    for i in range(batch_size):
        shape = insert_as_axis(input[i].shape, num_classes, axis, total_axes)
        result = np.zeros(shape, dtype=np.int32)
        shapes.append(shape)
        for i0 in range(input[i].shape[0]):
            for i1 in range(input[i].shape[1]):
                for i2 in range(input[i].shape[2]):
                    in_coord = (i0, i1, i2)
                    out_coord = insert_as_axis(in_coord, input[i][in_coord], axis, total_axes)
                    result[out_coord] = 1
        results.append(result)
    return results

def one_hot(input):
    outp = np.zeros([batch_size, num_classes], dtype=np.int32)
    for i in range(batch_size):
        outp[i, int(input[i])] = 1
    return outp



def check_one_hot_operator(premade_batch, axis=-1):
    pipeline = OneHotPipeline(num_classes=num_classes, input=premade_batch, axis=axis)
    pipeline.build()
    outputs = pipeline.run()
    sample_dim = len(premade_batch[0].shape)
    reference = one_hot_3_axes(premade_batch, axis) if sample_dim == 3 else one_hot(premade_batch)
    new_layout = None # TODO(klecki): add layout handling
    check_batch(outputs[0], reference, batch_size, max_allowed_error=0, expected_layout=new_layout)


def test_one_hot_scalar():
    np.random.seed(42)
    for i in range(10):
        premade_batch = np.random.randint(0, num_classes, size=batch_size, dtype=np.int32)
        yield check_one_hot_operator, premade_batch


def test_one_hot_legacy():
    np.random.seed(42)
    for i in range(10):
        premade_batch = [np.array([np.random.randint(0, num_classes)], dtype=np.int32)
                         for x in range(batch_size)]
        yield check_one_hot_operator, premade_batch


def test_one_hot ():
    np.random.seed(42)
    for i in range(10):
        for axis in [-1, 0, 1, 2, 3]:
            premade_batch = [
                np.random.randint(
                    0, num_classes, size=np.random.randint(2, 8, size=(3,)),
                    dtype=np.int32) for _ in range(batch_size)]
            yield check_one_hot_operator, premade_batch, axis
