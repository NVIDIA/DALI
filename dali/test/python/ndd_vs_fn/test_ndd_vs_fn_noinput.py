# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
from nvidia.dali.pipeline import pipeline_def

from nose2.tools import params
from ndd_vs_fn_test_utils import (
    MAX_BATCH_SIZE,
    OperatorTestConfig,
    compare_no_input,
    flatten_operator_configs,
)
from test_ndd_vs_fn_coverage import sign_off


NO_INPUT_OPERATORS = [
    OperatorTestConfig("transforms.translation", {"offset": (2, 3)}),
    OperatorTestConfig("transforms.scale", {"scale": (2, 3)}),
    OperatorTestConfig("transforms.rotation", {"angle": 30.0}),
    OperatorTestConfig("transforms.shear", {"shear": (2.0, 1.0)}),
    OperatorTestConfig(
        "transforms.crop",
        {
            "from_start": (0.0, 1.0),
            "from_end": (1.0, 1.0),
            "to_start": (0.2, 0.3),
            "to_end": (0.8, 0.5),
        },
    ),
    OperatorTestConfig("zeros", {"shape": (5, 5)}),
    OperatorTestConfig("ones", {"shape": (5, 5)}),
]

no_input_ops_test_configuration = flatten_operator_configs(NO_INPUT_OPERATORS)


@params(*no_input_ops_test_configuration)
def test_no_input_operators(device, operator_name, fn_operator, ndd_operator, operator_args):
    @pipeline_def(
        batch_size=MAX_BATCH_SIZE,
        num_threads=ndd.get_num_threads(),
        device_id=0,
        prefetch_queue_depth=1,
    )
    def pipeline():
        return fn_operator(device=device, **operator_args)

    pipe = pipeline()
    for _ in range(10):
        pipe_out = pipe.run()
        ndd_out = ndd_operator(device=device, batch_size=len(pipe_out[0]), **operator_args)
        compare_no_input(pipe_out, ndd_out)


@sign_off("transforms.combine")
def test_transforms_combine():
    @pipeline_def(
        batch_size=MAX_BATCH_SIZE,
        num_threads=ndd.get_num_threads(),
        device_id=0,
        prefetch_queue_depth=1,
    )
    def pipeline():
        t = fn.transforms.translation(offset=(1, 2))
        r = fn.transforms.rotation(angle=30.0)
        s = fn.transforms.scale(scale=(2, 3))
        out = fn.transforms.combine(t, r, s)
        return out

    pipe = pipeline()
    pipe.build()
    pipe_out = pipe.run()
    ndd_out = ndd.transforms.combine(
        ndd.transforms.translation(offset=(1, 2)),
        ndd.transforms.rotation(angle=30.0),
        ndd.transforms.scale(scale=(2, 3)),
        batch_size=MAX_BATCH_SIZE,
    )
    compare_no_input(pipe_out, ndd_out)
