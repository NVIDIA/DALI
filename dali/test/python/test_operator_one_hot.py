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

import numpy as np

sample_size = 10
premade_batch = [np.array(x) for x in range(sample_size)]


class OneHotPipeline(Pipeline):
    def __init__(self, nclasses, num_threads=1):
        super(OneHotPipeline, self).__init__(sample_size,
                                             num_threads,
                                             0)
        self.ext_src = ops.ExternalSource()
        self.one_hot = ops.OneHot(nclasses=nclasses, device="cpu")

    def define_graph(self):
        self.data = self.ext_src()
        return self.one_hot(self.data)

    def iter_setup(self):
        self.feed_input(self.data, premade_batch)


def one_hot(input):
    outp = np.zeros([sample_size, sample_size], dtype=int)
    for i in range(sample_size):
        outp[i, input[i]] = 1
    return outp


def check_one_hot_operator():
    pipeline = OneHotPipeline(nclasses=sample_size)
    pipeline.build()
    outputs = pipeline.run()
    reference = one_hot(premade_batch)
    outputs = outputs[0]
    for i in range(sample_size):
        output_row = outputs.at(i)
        for j in range(sample_size):
            # compare operator encoding with python implementation
            assert(output_row[j] == reference[i, j])


if __name__ == "__main__":
    check_one_hot_operator()
