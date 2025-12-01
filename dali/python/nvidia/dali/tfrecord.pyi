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

from typing import List

from typing import overload

float32: int
int64: int
string: int

class Feature:
    """
    TFRecord feature. Use the nvidia.dali.tfrecord.FixedLenFeature
    and nvidia.dali.tfrecord.VarLenFeature helper functions to obtain the instances.
    """

def FixedLenFeature(__shape: List[int], __dtype: int, __default_value: object, /) -> Feature:
    """Equivalent of TensorFlow's FixedLenFeature"""
    ...

@overload
def VarLenFeature(__dtype: int, __default_value: object, /) -> Feature:
    """Equivalent of TensorFlow's VarLenFeature"""
    ...

@overload
def VarLenFeature(__partial_shape: List[int], __dtype: int, __default_value: object, /) -> Feature:
    """Equivalent of TensorFlow's VarLenFeature"""
    ...
