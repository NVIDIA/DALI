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


class set_arithm_op_enabled:
    """Context-manager that enables arithmetic operators and slicing  on TensorLists.
    Can also be used as a function.
    """

    def __init__(self, mode=True):
        self.prev = set_arithm_op_enabled._arithm_op_enabled
        set_arithm_op_enabled._arithm_op_enabled = mode

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        set_arithm_op_enabled._arithm_op_enabled = self.prev

    _arithm_op_enabled = False


def is_arithm_op_enabled():
    return set_arithm_op_enabled._arithm_op_enabled
