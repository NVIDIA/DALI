# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


from numba import types

def _populate_setup_args(args_no, is_input_args=False):
    ret = []
    for _ in range(args_no):
        ret.append(types.CPointer(types.int64))
        ret.append(types.int32)
        ret.append(types.int32 if is_input_args else types.CPointer(types.int32))
    return ret

def setup_fn_sig(num_outputs, num_inputs):
    args_list = _populate_setup_args(num_outputs)
    args_list += _populate_setup_args(num_inputs, True)
    args_list.append(types.int32)
    return types.void(*args_list)

def _populate_run_args(types_list):
    ret = []
    types_list = [types_list] if not isinstance(types_list, list) else types_list
    for type in types_list:
        ret.append(types.CPointer(type))
        ret.append(types.CPointer(types.int64))
        ret.append(types.int32)
    return ret

def run_fn_sig(output_dtypes, input_dtypes):
    args_list = _populate_run_args(output_dtypes)
    args_list += _populate_run_args(input_dtypes)
    return types.void(*args_list)
