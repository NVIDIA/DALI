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

"""
DALI2 is a new experimental API that is currently under development.
"""

from enum import Enum, auto

class EvalMode(Enum):
    """Enum defining different evaluation modes for DALI2 operations.

    Attributes:
        default: Default evaluation mode
        lazy: Lazy evaluation mode - operations are evaluated only when their results are needed
        synchronous: Synchronous evaluation mode - operations are evaluated immediately
    """
    default = auto()
    lazy = auto()
    synchronous = auto()

