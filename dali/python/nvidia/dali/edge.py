# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#pylint: disable=no-name-in-module, unused-import
import nvidia.dali.backend
from nvidia.dali.backend import TensorCPU
from nvidia.dali.backend import TensorListCPU

print("[Warning] nvidia.dali.edge is deprecated. For TensorCPU and TensorListCPU " +
        "use nvidia.dali.tensors. EdgeReference is intended to be created only by " +
        "DALI Operators, and should not be instantiated directly by the user. " +
        "This class will be removed in DALI 0.17.")

# Deprecate and move to ops.py
class EdgeReference(object):
    def __init__(self, name, device="cpu", source=None):
        print("[Warning] EdgeReference from nvidia.dali.edge module is now deprecated. " +
                "This class is intended to be created only by DALI Operators, and should not be " +
                "instantiated directly by the user. This class will be removed in DALI 0.17.")
        self.name = name
        self.device = device
        self.source = source

    # Note: Regardless of whether we want the cpu or gpu version
    # of a tensor, we keep the source argument the same so that
    # the pipeline can backtrack through the user-defined graph
    def gpu(self):
        return EdgeReference(self.name, "gpu", self.source)
