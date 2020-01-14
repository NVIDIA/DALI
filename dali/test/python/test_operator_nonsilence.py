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
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np

sample_size = 10
batch_size = 5
premade_batch = [np.random.rand(sample_size) for i in range(batch_size)]


def nonsilence(buffer, cutoff_value):
    begin = 0
    end = 0
    for i in range(len(buffer)):
        if buffer[i] > cutoff_value:
            begin = i
            break
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i] > cutoff_value:
            end = i
            break
    length = 0 if begin == end else end - begin + 1
    return begin, length


class NonsilencePipeline(Pipeline):
    def __init__(self, cutoff_value, num_threads=1):
        super(NonsilencePipeline, self).__init__(batch_size, num_threads, 0)
        self.ext_src = ops.ExternalSource()
        self.nonsilence = ops.NonsilenceRegion(cutoff_value=cutoff_value, device="cpu")

    def define_graph(self):
        self.data = self.ext_src()
        return self.nonsilence(self.data)

    def iter_setup(self):
        self.feed_input(self.data, premade_batch)


def check_nonsilence_operator(cutoff_value):
    pipeline = NonsilencePipeline(cutoff_value=cutoff_value)
    pipeline.build()
    outputs = pipeline.run()
    for i in range(batch_size):
        ref_begin, ref_length = nonsilence(premade_batch[i], cutoff_value)
        assert ref_length == outputs[1].at(i)
        if ref_length > 0:
            assert ref_begin == outputs[0].at(i)


def test_nonsilence_operator():
    print(premade_batch)
    for coef in [.0, .5, 1.]:
        yield check_nonsilence_operator, coef
