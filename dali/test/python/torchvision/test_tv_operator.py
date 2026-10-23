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

import threading

import torch

from nose2.tools import params
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.experimental.torchvision.v2.operator import (
    adjust_input,
    is_sharing_random_values,
    share_random_values,
)


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
    probe(torch.zeros(3, 8, 8), device=device)
    assert seen == [device]


@params("cpu", "gpu")
def test_adjust_input_device_as_positional(device):
    probe, seen = make_device_probe()
    probe(torch.zeros(3, 8, 8), 1, device)
    assert seen == [device]


def test_adjust_input_device_defaults_to_cpu():
    probe, seen = make_device_probe()
    probe(torch.zeros(3, 8, 8))
    assert seen == ["cpu"]


def test_share_random_values_isolated_between_threads():
    # Pipelines defined concurrently in other threads must not see this pipeline's state
    seen_in_other_thread = []
    seen_in_pipeline = []

    @pipeline_def(batch_size=2, num_threads=1, device_id=None)
    def pipeline():
        input_node = fn.external_source(name="input_data")
        with share_random_values(input_node):
            thread = threading.Thread(
                target=lambda: seen_in_other_thread.append(is_sharing_random_values())
            )
            thread.start()
            thread.join()
            seen_in_pipeline.append(is_sharing_random_values())
        return input_node

    pipeline()
    assert seen_in_pipeline == [True]
    assert seen_in_other_thread == [False]
    assert not is_sharing_random_values()


def test_share_random_values_restores_outer_state():
    seen = []

    @pipeline_def(batch_size=2, num_threads=1, device_id=None)
    def pipeline():
        input_node = fn.external_source(name="input_data")
        with share_random_values(input_node):
            with share_random_values(input_node, enabled=False):
                seen.append(("inner", is_sharing_random_values()))
            seen.append(("outer", is_sharing_random_values()))
        return input_node

    pipeline()
    assert seen == [("inner", False), ("outer", True)]
    assert not is_sharing_random_values()
