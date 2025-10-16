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

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class ClahePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        num_threads=1,
        device_id=0,
        tiles_x=8,
        tiles_y=8,
        clip_limit=2.0,
        bins=256,
        luma_only=True,
        input_shape=(128, 128, 1),
    ):
        super(ClahePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.input_shape = input_shape
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.clip_limit = clip_limit
        self.bins = bins
        self.luma_only = luma_only

    def define_graph(self):
        # Create synthetic test data
        data = fn.random.uniform(range=(0, 255), shape=self.input_shape, dtype=types.UINT8)

        # Apply CLAHE
        if self.device == "gpu":
            data = data.gpu()

        clahe_output = fn.clahe(
            data,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            clip_limit=self.clip_limit,
            bins=self.bins,
            luma_only=self.luma_only,
        )

        return data, clahe_output


class ClaheOpsPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        num_threads=1,
        device_id=0,
        tiles_x=8,
        tiles_y=8,
        clip_limit=2.0,
        bins=256,
        luma_only=True,
        input_shape=(128, 128, 1),
    ):
        super(ClaheOpsPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.input_shape = input_shape

        self.clahe_op = ops.Clahe(
            device=device,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            clip_limit=clip_limit,
            bins=bins,
            luma_only=luma_only,
        )

    def define_graph(self):
        # Create synthetic test data
        data = fn.random.uniform(range=(0, 255), shape=self.input_shape, dtype=types.UINT8)

        if self.device == "gpu":
            data = data.gpu()

        clahe_output = self.clahe_op(data)

        return data, clahe_output


def test_clahe_operator_registration():
    """Test that CLAHE operator is properly registered."""
    # Check functional API
    assert hasattr(fn, "clahe"), "CLAHE operator not found in dali.fn"

    # Check class API
    assert hasattr(ops, "Clahe"), "CLAHE operator not found in dali.ops"

    # Check schema
    schema = dali.backend.TryGetSchema("Clahe")
    assert schema is not None, "CLAHE schema not found"
    assert schema.name == "Clahe"


def test_clahe_grayscale_gpu():
    """Test CLAHE with grayscale images on GPU."""
    batch_size = 4
    pipe = ClahePipeline(
        "gpu", batch_size, input_shape=(64, 64, 1), tiles_x=4, tiles_y=4, clip_limit=2.0
    )
    pipe.build()

    outputs = pipe.run()
    input_data, clahe_output = outputs

    # Verify output properties
    assert len(clahe_output) == batch_size
    for i in range(batch_size):
        original = np.array(input_data[i])
        enhanced = np.array(clahe_output[i])

        assert original.shape == enhanced.shape == (64, 64, 1)
        assert original.dtype == enhanced.dtype == np.uint8
        assert 0 <= enhanced.min() and enhanced.max() <= 255


def test_clahe_rgb_gpu():
    """Test CLAHE with RGB images on GPU."""
    batch_size = 2
    pipe = ClahePipeline(
        "gpu",
        batch_size,
        input_shape=(64, 64, 3),
        tiles_x=4,
        tiles_y=4,
        clip_limit=3.0,
        luma_only=True,
    )
    pipe.build()

    outputs = pipe.run()
    input_data, clahe_output = outputs

    # Verify output properties
    assert len(clahe_output) == batch_size
    for i in range(batch_size):
        original = np.array(input_data[i])
        enhanced = np.array(clahe_output[i])

        assert original.shape == enhanced.shape == (64, 64, 3)
        assert original.dtype == enhanced.dtype == np.uint8
        assert 0 <= enhanced.min() and enhanced.max() <= 255


def test_clahe_ops_api():
    """Test CLAHE using the ops API."""
    batch_size = 2
    pipe = ClaheOpsPipeline(
        "gpu", batch_size, input_shape=(32, 32, 1), tiles_x=2, tiles_y=2, clip_limit=1.5
    )
    pipe.build()

    outputs = pipe.run()
    input_data, clahe_output = outputs

    # Verify output properties
    assert len(clahe_output) == batch_size
    for i in range(batch_size):
        original = np.array(input_data[i])
        enhanced = np.array(clahe_output[i])

        assert original.shape == enhanced.shape == (32, 32, 1)
        assert original.dtype == enhanced.dtype == np.uint8


def test_clahe_parameter_validation():
    """Test parameter validation for CLAHE operator."""
    batch_size = 1

    # Valid parameters should work
    pipe = ClahePipeline("gpu", batch_size, tiles_x=8, tiles_y=8, clip_limit=2.0)
    pipe.build()

    # Test with different valid parameter combinations
    valid_configs = [
        {"tiles_x": 4, "tiles_y": 4, "clip_limit": 1.5},
        {"tiles_x": 16, "tiles_y": 8, "clip_limit": 3.0},
        {"tiles_x": 2, "tiles_y": 2, "clip_limit": 1.0},
    ]

    for config in valid_configs:
        pipe = ClahePipeline("gpu", batch_size, **config)
        pipe.build()
        outputs = pipe.run()
        assert len(outputs[1]) == batch_size


def test_clahe_different_tile_configurations():
    """Test CLAHE with different tile configurations."""
    batch_size = 2

    # Test different tile configurations
    tile_configs = [
        (2, 2),  # Few tiles
        (4, 4),  # Standard
        (8, 8),  # Many tiles
        (4, 8),  # Asymmetric
    ]

    for tiles_x, tiles_y in tile_configs:
        pipe = ClahePipeline(
            "gpu",
            batch_size,
            input_shape=(64, 64, 1),
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            clip_limit=2.0,
        )
        pipe.build()

        outputs = pipe.run()
        input_data, clahe_output = outputs

        # Verify all outputs are valid
        for i in range(batch_size):
            enhanced = np.array(clahe_output[i])
            assert enhanced.shape == (64, 64, 1)
            assert enhanced.dtype == np.uint8
