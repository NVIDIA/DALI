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

def _populate_args(types_list, is_input_args=False):
    ret = []
    for type in types_list:
        ret.append(types.CPointer(type))
        ret.append(types.int32)
        ret.append(types.int32 if is_input_args else types.CPointer(types.int32))
    return ret

def dali_numba_setup_fn_sig(num_outputs, output_dtypes, num_inputs, input_dtypes):
    assert num_outputs == len(output_dtypes)
    assert num_inputs == len(input_dtypes)
    
    args_list = _populate_args(output_dtypes)
    args_list += _populate_args(input_dtypes, True)
    for i in range(3):
        args_list.append(types.int32)
    return types.void(*args_list)