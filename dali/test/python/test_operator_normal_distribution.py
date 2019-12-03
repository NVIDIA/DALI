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

from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import scipy.stats as st


class NormalDistributionPipeline(Pipeline):
    def __init__(self, batch_size, num_threads=1):
        super(NormalDistributionPipeline, self).__init__(batch_size, num_threads, 0)


class NormalDistributionPipelineWithInput(NormalDistributionPipeline):
    def __init__(self, premade_batch):
        super(NormalDistributionPipelineWithInput, self).__init__(len(premade_batch))
        self.premade_batch = premade_batch
        self.ext_src = ops.ExternalSource()
        self.norm = ops.NormalDistribution(device="cpu", dtype=types.FLOAT)

    def define_graph(self):
        self.data = self.ext_src()
        return self.norm(self.data)

    def iter_setup(self):
        self.feed_input(self.data, self.premade_batch)


class NormalDistributionPipelineWithArgument(NormalDistributionPipeline):
    def __init__(self, shape):
        super(NormalDistributionPipelineWithArgument, self).__init__(1)
        # self.norm = ops.NormalDistribution(device="cpu", dtype=types.FLOAT)
        self.norm = ops.NormalDistribution(device="cpu", shape=shape, dtype=types.FLOAT)

    def define_graph(self):
        return self.norm()


class NormalDistributionPipelineDefault(NormalDistributionPipeline):
    def __init__(self, batch_size=1):
        super(NormalDistributionPipelineDefault, self).__init__(batch_size)
        self.norm = ops.NormalDistribution(device="cpu", dtype=types.FLOAT)

    def define_graph(self):
        return self.norm()


def test_normal_distribution_with_input():
    premade_batches = [[np.random.rand(1000), np.random.rand(10000)],
                       [np.random.rand(100, 100), np.random.rand(100, 100)],
                       [np.random.rand(100, 10, 10), np.random.rand(100, 10, 10)]]
    for batch in premade_batches:
        bsize = len(batch)
        pipeline = NormalDistributionPipelineWithInput(premade_batch=batch)
        pipeline.build()
        outputs = pipeline.run()
        for i in range(bsize):
            print(outputs[0].at(i))
            print(outputs[0].at(i).shape)
            assert outputs[0].at(i).shape == batch[i].shape
            possibly_normal_distribution = outputs[0].at(i).flatten()
            _, pvalues_anderson, _ = st.anderson(possibly_normal_distribution, dist='norm')
            # It's not 100% mathematically correct, but makes do in case of this test
            assert pvalues_anderson[2] > 0.5


def test_normal_distribution_with_argument():
    pipeline = NormalDistributionPipelineWithArgument(shape=[1])
    pipeline.build()
    # outputs = pipeline.run()
    # print(outputs[0].at(0))
    # print(outputs[0].at(0).shape)
    # possibly_normal_distribution = outputs[0].at(0).flatten()
    # _, pvalues_anderson, _ = st.anderson(possibly_normal_distribution, dist='norm')
    # It's not 100% mathematically correct, but makes do in case of this test
    # assert pvalues_anderson[2] > 0.5


def test_normal_distribution_default():
    pipeline = NormalDistributionPipelineDefault(100)
    pipeline.build()
    outputs = pipeline.run()
    possibly_normal_distribution = outputs[0].as_array().flatten()
    _, pvalues_anderson, _ = st.anderson(possibly_normal_distribution, dist='norm')
    # It's not 100% mathematically correct, but makes do in case of this test
    assert pvalues_anderson[2] > 0.5


# test_normal_distribution_with_input()
test_normal_distribution_with_argument()
# test_normal_distribution_default()
