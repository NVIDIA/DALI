# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.backend as _b
from packaging.version import Version

# Python-based operators are not supported in dynamic mode
PythonBasedOps = ["ExternalSource", "PythonFunction", "NumbaFunction", "JaxFunction"]

# Operators that were deprecated before this version will not be exposed in dynamic mode
DynamicModeOpCutoff = Version("2.0")


def should_create_dynamic_op(schema_name: str) -> bool:
    """
    Determines if an operator with the given schema name should be exposed in dynamic mode.
    """
    if any(schema_name.endswith(op) for op in PythonBasedOps):
        return False
    schema = _b.GetSchema(schema_name)
    if schema.IsInternal():
        return False
    if schema.IsDeprecated():
        deprecated_version = Version(schema.DeprecatedInVersion())
        if deprecated_version <= DynamicModeOpCutoff:
            return False
    return True
