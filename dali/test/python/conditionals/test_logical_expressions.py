

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali.pipeline import pipeline_def, experimental
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import SampleInfo
from nvidia.dali import _conditionals
from nvidia.dali.data_node import DataNode

from test_utils import check_batch, compare_pipelines
from nose_utils import assert_raises
from test_utils import get_dali_extra_path
from nose2.tools import params

import os


from nvidia.dali._autograph.utils import ag_logging

ag_logging.set_verbosity(10, True)


def test():
    test_data_root = get_dali_extra_path()
    caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @experimental.pipeline_def(enable_conditionals=True, **kwargs)
    def conditional_pipe():
        val = fn.random.uniform(values=[0, 1, 2])
        x = val and (not val.gpu() == 1)
        return x

    pipe = conditional_pipe()
    pipe.build()
    pipe.save_graph_to_dot_file("or.dot")
    pipe.save_graph_to_dot_file("or_full.dot", True, True, True)
    pipe.run()