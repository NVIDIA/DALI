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
    feed_input,
    pipeline_es_feed_input_wrapper,
    compare_no_input,
    flatten_operator_configs,
    generate_image_like_data,
)
from test_ndd_vs_fn_coverage import register_operator_test


NO_INPUT_OPERATORS = [
    OperatorTestConfig("transforms.translation", {"offset": (2, 3)}, devices=["cpu"]),
    OperatorTestConfig("transforms.scale", {"scale": (2, 3)}, devices=["cpu"]),
    OperatorTestConfig("transforms.rotation", {"angle": 30.0}, devices=["cpu"]),
    OperatorTestConfig("transforms.shear", {"shear": (2.0, 1.0)}, devices=["cpu"]),
    OperatorTestConfig(
        "transforms.crop",
        {
            "from_start": (0.0, 1.0),
            "from_end": (1.0, 1.0),
            "to_start": (0.2, 0.3),
            "to_end": (0.8, 0.5),
        },
        devices=["cpu"],
    ),
    OperatorTestConfig("zeros", {"shape": (5, 5)}, devices=["cpu"]),
    OperatorTestConfig("ones", {"shape": (5, 5)}, devices=["cpu"]),
]

no_input_ops_test_configuration = flatten_operator_configs(NO_INPUT_OPERATORS)


@params(*no_input_ops_test_configuration)
def test_no_input_operators(device, operator_name, fn_operator, ndd_operator, operator_args):
    register_operator_test(operator_name)
    data = generate_image_like_data()
    # Passing input to no-input operator is artificial,
    # and it's here to avoid pruning no-input operator from the graph.
    pipe = pipeline_es_feed_input_wrapper(
        fn_operator,
        device,
        input_layout=None,
        num_inputs=1,
        needs_input=False,
        **operator_args,
    )
    for inp in data:
        feed_input(pipe, inp)
        pipe_out = pipe.run()
        ndd_out = ndd_operator(device=device, batch_size=inp.shape[0], **operator_args)
        compare_no_input(pipe_out, ndd_out)


def test_transforms_combine():
    register_operator_test("transforms.combine")

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
