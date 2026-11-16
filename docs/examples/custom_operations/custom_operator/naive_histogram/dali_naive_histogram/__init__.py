#  Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Importing this package loads the NaiveHistogram plugin (``fn.naive_histogram``)."""

import os

import nvidia.dali as dali
import nvidia.dali.plugin_manager as plugin_manager

from ._build_info import DALI_VERSION as _BUILT_WITH_DALI_VERSION

# A plugin can only be loaded by the exact DALI version it was built against.
if dali.__version__ != _BUILT_WITH_DALI_VERSION:
    raise RuntimeError(
        f"dali_naive_histogram was built against DALI {_BUILT_WITH_DALI_VERSION}, but DALI "
        f"{dali.__version__} is installed. Rebuild the plugin for the installed DALI version: "
        "pip install --no-build-isolation --no-cache-dir --force-reinstall <path to the sdist>"
    )

plugin_manager.load_library(
    os.path.join(os.path.dirname(__file__), "libnaivehistogram.so")
)
