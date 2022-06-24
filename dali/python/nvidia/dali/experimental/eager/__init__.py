# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import math  # noqa: F401

""" Eager module implements eager versions of standard DALI operators.
There are 3 main types of eager operators:
    stateless: callable directly as functions. Example:
        >> dali.experimental.eager.crop(input, crop=(5, 5))
    stateful: operators that require a state, used as method of the `rng_state`. Example:
        >> rng_state = dali.experimental.eager.rng_state(seed=42)
        >> rng_state.random.normal(shape=(10, 10), batch_size=8))
    iterators: reader operators as python iterables. Example:
        >> for file, label in eager.readers.file(file_root=file_path, batch_size=8):
        ..     # file and label are batches of size 8 (TensorLists).
        ..     print(file)
Additionally eager implements:
    math operators - `dali.experimental.eager.math`. Example:
        >> tl = dali.tensors.TensorListCPU(...)
        >> dali.experimental.eager.math.sqrt(tl)
    direct arithmetic operators on TensorLists, enabled with `dali.experimental.eager.arithmetic`
        as context-manager or as function with global setting.
"""


class _MetaArithmetic(type):
    @property
    def enabled(cls):
        return cls._enabled


class arithmetic(metaclass=_MetaArithmetic):
    """ Context-manager that enabled/disables arithmetic operators on TensorLists.
    Can also be used as a function with global setting.

    Examples:
        >> tl = dali.tensors.TensorListCPU(...)
        >> with dali.experimental.eager.arithmetic(enabled=True):
        ..     out = tl + 1

        >> dali.experimental.eager.arithmetic(enabled=True)
        >> tl = dali.tensors.TensorListCPU(...)
        >> out = tl ** 2
    """
    def __init__(self, enabled=True):
        self.prev = arithmetic._enabled
        arithmetic._enabled = enabled

    @property
    def enabled(self):
        return type(self)._enabled

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        arithmetic._enabled = self.prev

    _enabled = False
