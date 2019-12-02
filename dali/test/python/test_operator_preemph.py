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

sample_size = 100
premade_batch = [np.random.rand(sample_size)]


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Python2.7 doesn't have math.isclose()
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def preemph(signal, coeff=0.0):
    ret = np.copy(signal)
    if coeff == 0.0:
        return ret
    for i in range(sample_size - 1, 0, -1):
        ret[i] -= coeff * ret[i - 1]
    ret[0] -= coeff * ret[0]
    return ret


class PreemphasisPipeline(Pipeline):
    def __init__(self, preemph_coeff=0., num_threads=1):
        super(PreemphasisPipeline, self).__init__(1, num_threads, 0)
        self.ext_src = ops.ExternalSource()
        self.preemph = ops.PreemphasisFilter(preemph_coeff=preemph_coeff, device="cpu")

    def define_graph(self):
        self.data = self.ext_src()
        return self.preemph(self.data)

    def iter_setup(self):
        self.feed_input(self.data, premade_batch)


def check_preemphasis_operator(preemph_coeff):
    eps = 1e-5
    pipeline = PreemphasisPipeline(preemph_coeff=preemph_coeff)
    pipeline.build()
    outputs = pipeline.run()
    reference_signal = preemph(premade_batch[0], preemph_coeff)
    for i in range(sample_size):
        a = outputs[0].at(0)[i]
        b = reference_signal[i]
        assert isclose(a, b, abs_tol=eps)


def test_preemphasis_operator():
    for coef in [0.5, 0.0]:
       yield check_preemphasis_operator, coef
