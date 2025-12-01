# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Global configuration."""

from nvidia.dali._autograph.core import config_lib

Action = config_lib.Action
Convert = config_lib.Convert
DoNotConvert = config_lib.DoNotConvert


# This list is evaluated in order and stops at the first rule that tests True
# for a definitely_convert of definitely_bypass call.
# The name under which autograph is used is registered in init_autograph.
CONVERSION_RULES = (
    # Known packages
    # <List of packages that we force-convert>
    # Builtin modules
    DoNotConvert("collections"),
    DoNotConvert("copy"),
    DoNotConvert("cProfile"),
    DoNotConvert("inspect"),
    DoNotConvert("ipdb"),
    DoNotConvert("linecache"),
    DoNotConvert("mock"),
    DoNotConvert("pathlib"),
    DoNotConvert("pdb"),
    DoNotConvert("posixpath"),
    DoNotConvert("pstats"),
    DoNotConvert("re"),
    DoNotConvert("threading"),
    DoNotConvert("urllib"),
    # Known libraries
    DoNotConvert("matplotlib"),
    DoNotConvert("numpy"),
    DoNotConvert("pandas"),
    DoNotConvert("PIL"),
    DoNotConvert("absl.logging"),
)
