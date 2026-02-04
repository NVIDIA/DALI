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


from nose2.tools import params
from ndd_vs_fn_test_utils import (
    OperatorTestConfig,
    feed_input,
    pipeline_es_feed_input_wrapper,
    compare_no_input,
    flatten_operator_configs,
    generate_image_like_data,
)


NO_INPUT_OPERATORS = [
    OperatorTestConfig("constant", {"fdata": 3.1415, "shape": (10, 10)}),
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
def test_no_input_operators(device, fn_operator, ndd_operator, operator_args):
    data = generate_image_like_data()
    # Passing input to no-input operator is artificial,
    # and it's here to avoid prunning no-input operator from the graph.
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
        assert compare_no_input(pipe_out, ndd_out)


tested_operators = [
    "constant",
    "transforms.translation",
    "transforms.scale",
    "transforms.rotation",
    "transforms.shear",
    "transforms.crop",
    "zeros",
    "ones",
]
