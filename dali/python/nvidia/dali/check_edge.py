# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

def _validate_edge_reference(edge):
    # TODO(klecki): adjust to <class 'nvidia.dali.ops._EdgeReference'>
    if not "EdgeReference" in str(type(edge)):
        raise TypeError(("Expected outputs of type compatible with \"EdgeReference\"."
                " Received output type with name \"{}\" that does not match.")
                .format(type(edge).__name__))
    for attr in ["name", "device", "source"]:
      if not hasattr(edge, attr):
          raise TypeError(("Expected outputs of type compatible with \"EdgeReference\". Received"
                  " output type \"{}\" does not have the attribute \"{}\" that is required.")
                  .format(type(edge).__name__, attr))
    return True
