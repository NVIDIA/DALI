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

import sys

pipeline_serialized = False


def autoserialize(dali_pipeline):
    """
    TODO
    :param dali_pipeline:
    :return:
    """
    if len(sys.argv) != 3 or sys.argv[2] != "autoserialize.me":
        return
    global pipeline_serialized
    assert not pipeline_serialized, f"There can be only one autoserialized pipeline in a file. Offending pipeline name: {dali_pipeline.__qualname__}"
    filepath = sys.argv[1]
    dali_pipeline().serialize(filename=filepath)
    pipeline_serialized = True
