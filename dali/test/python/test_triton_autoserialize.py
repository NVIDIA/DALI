# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._utils.autoserialize import invoke_autoserialize
from nose_utils import raises

serialized_filename = "/tmp/some_custom_name"


def test_direct_import():
    from autoserialize_test import decorated_function

    invoke_autoserialize(decorated_function, serialized_filename)


def test_indirect_import():
    from autoserialize_test import imports_decorated_function

    invoke_autoserialize(imports_decorated_function, serialized_filename)


@raises(RuntimeError, glob="Precisely one autoserialize function must exist in the module.*")
def test_double_decorated_functions():
    from autoserialize_test import double_decorated_functions

    invoke_autoserialize(double_decorated_functions, serialized_filename)


@raises(TypeError, glob="Only `@pipeline_def` can be decorated with `@triton.autoserialize`.")
def test_improper_decorated_function():
    from autoserialize_test import improper_decorated_function

    invoke_autoserialize(improper_decorated_function, serialized_filename)


def test_custom_module():
    from autoserialize_test import custom_module_inside

    invoke_autoserialize(custom_module_inside, serialized_filename)
