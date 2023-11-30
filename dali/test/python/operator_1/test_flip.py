# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os

from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")


class FlipPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        is_vertical=0,
        is_horizontal=1,
    ):
        super(FlipPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.input = ops.readers.Caffe(
            path=caffe_db_folder, shard_id=device_id, num_shards=num_gpus
        )
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        self.flip = ops.Flip(device=self.device, vertical=is_vertical, horizontal=is_horizontal)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == "gpu":
            images = images.gpu()
        images = self.flip(images)
        return images


class SynthFlipPipeline(Pipeline):
    def __init__(self, batch_size, layout, data_iterator, device):
        super(SynthFlipPipeline, self).__init__(batch_size, seed=1234, num_threads=4, device_id=0)
        self.device = device
        self.iterator = data_iterator
        self.layout = layout
        self.input = ops.ExternalSource()
        self.coin = ops.random.CoinFlip(seed=1234)
        self.flip = ops.Flip(device=device)

    def define_graph(self):
        self.data = self.input()
        data = self.data.gpu() if self.device == "gpu" else self.data
        flipped = self.flip(
            data, horizontal=self.coin(), vertical=self.coin(), depthwise=self.coin()
        )

        return flipped

    def iter_setup(self):
        self.feed_input(self.data, self.iterator.next(), layout=self.layout)


def numpy_flip(data, h_dim, v_dim, d_dim, hor, ver, depth):
    if h_dim >= 0 and hor:
        data = np.flip(data, h_dim)
    if v_dim >= 0 and ver:
        data = np.flip(data, v_dim)
    if d_dim >= 0 and depth:
        data = np.flip(data, d_dim)
    return data


def find_dims(layout):
    return layout.find("W"), layout.find("H"), layout.find("D")


class SynthPythonFlipPipeline(Pipeline):
    def __init__(self, batch_size, layout, data_iterator):
        super(SynthPythonFlipPipeline, self).__init__(
            batch_size,
            seed=1234,
            num_threads=4,
            device_id=0,
            exec_async=False,
            exec_pipelined=False,
        )
        self.iterator = data_iterator
        self.layout = layout
        self.input = ops.ExternalSource()
        self.coin = ops.random.CoinFlip(seed=1234)
        h_dim, v_dim, d_dim = find_dims(layout)

        def fun(d, hor, ver, depth):
            return numpy_flip(d, h_dim, v_dim, d_dim, hor, ver, depth)

        self.python_flip = ops.PythonFunction(function=fun, output_layouts=layout)

    def define_graph(self):
        self.data = self.input()
        flipped = self.python_flip(self.data, self.coin(), self.coin(), self.coin())
        return flipped

    def iter_setup(self):
        self.feed_input(self.data, self.iterator.next(), layout=self.layout)


def check_flip(batch_size, layout, shape, device):
    eiis = [RandomDataIterator(batch_size, shape=shape) for k in range(2)]
    compare_pipelines(
        SynthFlipPipeline(batch_size, layout, iter(eiis[0]), device),
        SynthPythonFlipPipeline(batch_size, layout, iter(eiis[1])),
        batch_size=batch_size,
        N_iterations=3,
    )


def test_flip_vs_numpy():
    for batch_size in [1, 8, 32]:
        for device in ["cpu", "gpu"]:
            for layout, shape in [
                ("HWC", (15, 20, 3)),
                ("CHW", (4, 20, 25)),
                ("DHWC", (10, 20, 30, 2)),
                ("CDHW", (2, 5, 10, 15)),
                ("FHWC", (3, 90, 120, 3)),
                ("FCHW", (4, 3, 100, 150)),
                ("FDHWC", (4, 20, 50, 30, 3)),
                ("FCDHW", (3, 3, 20, 50, 30)),
            ]:
                yield check_flip, batch_size, layout, shape, device
