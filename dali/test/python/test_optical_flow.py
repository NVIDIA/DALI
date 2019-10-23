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
from __future__ import division
import os
import numpy as np
import shutil
import sys
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from random import shuffle
from test_utils import get_dali_extra_path

batch_size = 1
sequence_length = 2
dali_extra_path = get_dali_extra_path()
image_dir = os.path.join(dali_extra_path, "db", "optical_flow", "slow_preset", "two_frames")


class OFPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(OFPipeline, self).__init__(batch_size, num_threads, device_id, seed=16)
        self.input = ops.SequenceReader(file_root=image_dir, sequence_length=sequence_length)
        self.of_op = ops.OpticalFlow(device="gpu", output_format=4)
    def define_graph(self):
        seq = self.input(name="Reader")
        of = self.of_op(seq.gpu())
        return of


def test_of():
    pipe = OFPipeline(batch_size=batch_size, num_threads=1, device_id=0)
    pipe.build()
    pipe_out = pipe.run()
    frames = pipe_out[0].as_cpu().as_array()
    myarray = np.loadtxt(os.path.join(dali_extra_path, "db", "optical_flow", "slow_preset",
                                      "decoded_flow_vector.dat"))
    assert (0.9 > np.mean(np.abs(frames[0][0].flatten() - myarray)))

