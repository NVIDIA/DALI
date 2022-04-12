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

from nvidia.dali._utils import invoke_autoserialize
from nose import raises

serialized_filename = "/tmp/some_custom_name"


def test_direct_import():
    from .triton import decorated_function
    invoke_autoserialize(decorated_function, serialized_filename)


def test_not_direct_import():
    from .triton import imports_decorated_function
    invoke_autoserialize(imports_decorated_function, serialized_filename)


@raises
def test_double_decorated_functions():
    from .triton import double_decorated_functions
    invoke_autoserialize(double_decorated_functions, serialized_filename)


@raises
def test_improper_decorated_function():
    from .triton import improper_decorated_function
    invoke_autoserialize(improper_decorated_function, serialized_filename)
