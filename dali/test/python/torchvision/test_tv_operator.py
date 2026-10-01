# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from nose2.tools import params
from nvidia.dali.experimental.torchvision.v2.operator import adjust_input


def make_device_probe():
    seen = []

    @adjust_input
    def probe(inpt, scale: int = 1, device="cpu"):
        """Probe docstring."""
        seen.append(inpt.device.device_type)
        return inpt

    return probe, seen


def test_adjust_input_preserves_metadata():
    probe, _ = make_device_probe()
    assert probe.__name__ == "probe"
    assert probe.__doc__ == "Probe docstring."


@params("cpu", "gpu")
def test_adjust_input_device_as_keyword(device):
    probe, seen = make_device_probe()
    probe(torch.rand(3, 8, 8), device=device)
    assert seen == [device]


@params("cpu", "gpu")
def test_adjust_input_device_as_positional(device):
    probe, seen = make_device_probe()
    probe(torch.rand(3, 8, 8), 1, device)
    assert seen == [device]


def test_adjust_input_device_defaults_to_cpu():
    probe, seen = make_device_probe()
    probe(torch.rand(3, 8, 8))
    assert seen == ["cpu"]
