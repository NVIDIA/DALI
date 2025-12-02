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
Dynamic API - a new experimental API that allows to interleave DALI operations with Python code.
"""

from ._eval_mode import *  # noqa: F401, F403
from ._eval_context import *  # noqa: F401, F403
from ._type import *  # noqa: F401, F403
from ._device import *  # noqa: F401, F403
from ._tensor import Tensor, tensor, as_tensor  # noqa: F401
from ._batch import Batch, batch, as_batch  # noqa: F401
from ._imread import imread  # noqa: F401

from . import ops
from . import math  # noqa: F401
from . import random  # noqa: F401

ops._initialize()
