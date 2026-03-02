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
    run_operator_test,
    flatten_operator_configs,
    array_1d_shape_generator,
    generate_data,
)


ARRAY_1D_OPERATORS = [
    OperatorTestConfig("power_spectrum"),
    OperatorTestConfig("preemphasis_filter"),
    OperatorTestConfig("spectrogram", {"nfft": 60, "window_length": 50, "window_step": 25}),
    OperatorTestConfig("to_decibels"),
    OperatorTestConfig("audio_resample", {"scale": 1.2}),
    OperatorTestConfig("one_hot", {"on_value": 0.5}),
]

ops_1d_float_array_test_configuration = flatten_operator_configs(ARRAY_1D_OPERATORS)


@params(*ops_1d_float_array_test_configuration)
def test_operators_with_array_1d_input(
    device, operator_name, fn_operator, ndd_operator, operator_args
):
    data = generate_data(array_1d_shape_generator)
    run_operator_test(
        input_epoch=data,
        fn_operator=fn_operator,
        ndd_operator=ndd_operator,
        device=device,
        operator_args=operator_args,
    )
