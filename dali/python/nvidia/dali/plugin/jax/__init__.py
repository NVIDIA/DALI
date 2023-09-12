# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import sys
import jax

from distutils.version import LooseVersion
from .iterator import DALIGenericIterator

assert sys.version_info.major == 3 and sys.version_info.minor >= 8, \
    "DALI JAX support requires Python 3.8 or above"


assert LooseVersion(jax.__version__) >= LooseVersion('0.4.11'), \
    "DALI JAX support requires JAX 0.4.11 or above"


__all__ = ["DALIGenericIterator"]
