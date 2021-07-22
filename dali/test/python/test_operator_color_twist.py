# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from test_utils import RandomDataIterator

class ColorTwistPipeline(Pipeline):
    def __init__(self, batch_size, seed, data_iterator, kind="new", num_threads=1, device_id=0):
        super(ColorTwistPipeline, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.input = ops.ExternalSource(source=data_iterator)
        self.hue = ops.random.Uniform(range=[-20., 20.], seed=seed)
        self.sat = ops.random.Uniform(range=[0., 1.], seed=seed)
        self.bri = ops.random.Uniform(range=[0., 2.], seed=seed)
        self.con = ops.random.Uniform(range=[0., 2.], seed=seed)
        self.kind = kind
        if kind == "new":
            self.color_twist = ops.ColorTwist(device="gpu")
        elif kind == "old":
            self.color_twist = ops.OldColorTwist(device="gpu")
        else:
            self.color_twist = ops.OldColorTwist(device="cpu")

    def define_graph(self):
        self.images = self.input()
        imgs = self.images if self.kind == "oldCpu" else self.images.gpu()
        hue = self.hue() if self.kind in ["old", "oldCpu"] else -self.hue()
        return self.color_twist(imgs, hue=hue, saturation=self.sat(), brightness=self.bri(), contrast=self.con())


def test_color_twist_vs_old():
    batch_size = 16
    seed = 2139
    rand_it1 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    rand_it2 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    compare_pipelines(ColorTwistPipeline(batch_size, seed, iter(rand_it1), kind="new"),
                      ColorTwistPipeline(batch_size, seed, iter(rand_it2), kind="old"),
                      batch_size=batch_size, N_iterations=3, eps=1)


def test_color_twist_vs_cpu():
    batch_size = 8
    seed = 1919
    rand_it1 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    rand_it2 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    compare_pipelines(ColorTwistPipeline(batch_size, seed, iter(rand_it1), kind="new"),
                      ColorTwistPipeline(batch_size, seed, iter(rand_it2), kind="oldCpu"),
                      batch_size=batch_size, N_iterations=3, eps=1)
