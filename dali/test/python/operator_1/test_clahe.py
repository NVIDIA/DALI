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
from nvidia.dali import fn, ops, types
from nvidia.dali.pipeline import Pipeline
import cv2

MSE_THRESHOLD = 5.0
MAE_THRESHOLD = 2.0


def create_synthetic_test_images():
    """Create synthetic test images good for CLAHE testing"""
    test_images = {}

    # 1. Low contrast gradient image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            val = int(50 + 50 * np.sin(i * 0.02) * np.cos(j * 0.02))
            img[i, j] = [val, val, val]
    test_images["low_contrast_gradient"] = img

    # 2. High contrast checkerboard with poor local contrast
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    block_size = 32
    for i in range(0, 256, block_size):
        for j in range(0, 256, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                # Dark block with some detail
                img[i : i + block_size, j : j + block_size] = np.random.randint(
                    20, 40, (block_size, block_size, 3)
                )
            else:
                # Bright block with some detail
                img[i : i + block_size, j : j + block_size] = np.random.randint(
                    200, 220, (block_size, block_size, 3)
                )
    test_images["high_contrast_blocks"] = img

    # 3. Medical/X-ray like image
    img = np.zeros((256, 256, 1), dtype=np.uint8)
    center = (128, 128)
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            val = int(30 + 60 * np.exp(-dist / 50) + 20 * np.random.random())
            val = np.clip(val, 0, 255)
            img[i, j] = val
    test_images["medical_scan"] = img

    return test_images


def apply_opencv_clahe(image, tiles_x=8, tiles_y=8, clip_limit=2.0, luma_only=True):
    """Apply OpenCV CLAHE to an image with enhanced precision"""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tiles_x, tiles_y))

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            # Single channel image with 3D shape (H, W, 1) - treat as grayscale
            result = clahe.apply(image[:, :, 0])
            result = np.expand_dims(result, axis=2)  # Keep 3D shape
        elif luma_only and image.shape[2] == 3:
            # RGB image - apply to luminance channel only
            # Use LAB color space to match DALI exactly
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            # Apply CLAHE to L channel
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            # Convert back to RGB
            result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        elif image.shape[2] == 3:
            # Apply CLAHE to each RGB channel separately
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = clahe.apply(image[:, :, i])
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        # Grayscale
        result = clahe.apply(image)

    return result


class MemoryPipeline(Pipeline):
    """DALI pipeline using external data input for exact comparison"""

    def __init__(self, image_array, tiles_x=8, tiles_y=8, clip_limit=2.0, device="gpu"):
        super().__init__(batch_size=1, num_threads=1, device_id=0)
        self.image_array = image_array
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.clip_limit = clip_limit
        self.device = device

    def define_graph(self):
        # Use external source to feed exact same data as OpenCV
        images = fn.external_source(
            source=lambda: [self.image_array],
            device="cpu",
            ndim=len(self.image_array.shape),
        )

        if self.device == "gpu":
            # Move to GPU for processing
            images_processed = images.gpu()
        else:
            # Keep on CPU for processing
            images_processed = images

        # Apply CLAHE operator
        clahe_result = fn.clahe(
            images_processed,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            clip_limit=float(self.clip_limit),
            luma_only=True,
            device=self.device,
        )

        return clahe_result


def apply_dali_clahe_from_memory(image_array, tiles_x=8, tiles_y=8, clip_limit=2.0, device="gpu"):
    """Apply DALI CLAHE using memory-based pipeline for exact input matching"""
    try:
        # Create memory-based pipeline
        pipe = MemoryPipeline(image_array, tiles_x, tiles_y, clip_limit, device)
        pipe.build()

        # Run pipeline
        outputs = pipe.run()
        result = outputs[0].as_cpu().as_array()[0]  # Get first image from batch

        # Enhanced data type conversion with rounding for better precision
        if result.dtype != np.uint8:
            # Round to nearest integer before clipping for better accuracy
            result = np.round(np.clip(result, 0, 255)).astype(np.uint8)

        return result

    except Exception as e:
        print(f"DALI CLAHE ({device}) failed: {e}")
        return None


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
        # Create synthetic test data - CLAHE requires uint8 input
        data = fn.cast(
            fn.random.uniform(range=(0, 255), shape=self.input_shape, seed=816),
            dtype=types.DALIDataType.UINT8,
        )

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
        # Create synthetic test data - CLAHE requires uint8 input
        data = fn.cast(
            fn.random.uniform(range=(0, 255), shape=self.input_shape, seed=816),
            dtype=types.DALIDataType.UINT8,
        )

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

    # Check schema (simplified check without backend access)
    try:
        # Try to create an instance which will verify the operator exists
        test_op = ops.Clahe(device="cpu")
        assert test_op is not None, "CLAHE operator could not be instantiated"
    except Exception as e:
        assert False, f"CLAHE operator registration failed: {e}"


def test_clahe_grayscale_gpu():
    """Test CLAHE with grayscale images on GPU."""
    input_shapes = [
        (256, 256, 1),
        (128, 128, 1),
        (64, 64, 1),
    ]
    for batch_size in [1, 4, 8]:
        for input_shape in input_shapes:
            pipe = ClahePipeline(
                "gpu",
                batch_size,
                input_shape=input_shape,
                tiles_x=4,
                tiles_y=4,
                clip_limit=2.0,
            )
            pipe.build()

            outputs = pipe.run()
            input_data, clahe_output = outputs

            # Verify output properties
            assert len(clahe_output) == batch_size
            for i in range(batch_size):
                original = np.array(input_data[i].as_cpu())
                enhanced = np.array(clahe_output[i].as_cpu())

                assert original.shape == enhanced.shape == input_shape
                assert original.dtype == enhanced.dtype == np.uint8
                assert 0 <= enhanced.min() and enhanced.max() <= 255


def test_clahe_rgb_gpu():
    """Test CLAHE with RGB images on GPU."""
    input_shapes = [
        (64, 64, 3),
        (128, 128, 3),
        (32, 32, 3),
    ]
    for batch_size in [1, 4]:
        for input_shape in input_shapes:
            pipe = ClahePipeline(
                "gpu",
                batch_size,
                input_shape=input_shape,
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
                original = np.array(input_data[i].as_cpu())
                enhanced = np.array(clahe_output[i].as_cpu())

                assert original.shape == enhanced.shape == input_shape
                assert original.dtype == enhanced.dtype == np.uint8
                assert 0 <= enhanced.min() and enhanced.max() <= 255


def test_clahe_ops_api():
    """Test CLAHE using the ops API."""
    input_shapes = [
        (32, 32, 1),
        (64, 64, 1),
        (32, 32, 3),
    ]

    for batch_size in [2, 5]:
        for input_shape in input_shapes:
            pipe = ClaheOpsPipeline(
                "gpu",
                batch_size,
                input_shape=input_shape,
                tiles_x=2,
                tiles_y=2,
                clip_limit=1.5,
            )
            pipe.build()

            outputs = pipe.run()
            input_data, clahe_output = outputs

            # Verify output properties
            assert len(clahe_output) == batch_size
            for i in range(batch_size):
                original = np.array(input_data[i].as_cpu())
                enhanced = np.array(clahe_output[i].as_cpu())

                assert original.shape == enhanced.shape == input_shape
                assert original.dtype == enhanced.dtype == np.uint8


def test_clahe_parameter_validation():
    """Test parameter validation for CLAHE operator."""

    for batch_size in [1, 4]:
        # Valid parameters should work
        pipe = ClahePipeline("gpu", batch_size, tiles_x=8, tiles_y=8, clip_limit=2.0)
        pipe.build()

        # Test with different valid parameter combinations
        valid_configs = [
            {"tiles_x": 4, "tiles_y": 4, "clip_limit": 1.5},
            {"tiles_x": 8, "tiles_y": 8, "clip_limit": 2.0},
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
            enhanced = np.array(clahe_output[i].as_cpu())
            assert enhanced.shape == (64, 64, 1)
            assert enhanced.dtype == np.uint8


def test_clahe_opencv_comparison_gpu():
    """Test CLAHE GPU implementation against OpenCV with MSE/MAE assertions."""
    test_images = create_synthetic_test_images()

    for test_name, test_image in test_images.items():
        # Apply OpenCV CLAHE
        opencv_result = apply_opencv_clahe(test_image, tiles_x=4, tiles_y=4, clip_limit=2.0)

        # Apply DALI CLAHE GPU
        dali_result = apply_dali_clahe_from_memory(
            test_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="gpu"
        )

        assert dali_result is not None, f"DALI GPU CLAHE failed for {test_name}"

        # Calculate metrics
        opencv_float = opencv_result.astype(np.float64)
        dali_float = dali_result.astype(np.float64)

        mse = np.mean((opencv_float - dali_float) ** 2)
        mae = np.mean(np.abs(opencv_float - dali_float))

        # Assert MSE and MAE are under reasonable thresholds
        # Different LAB implementations can have notable differences, especially for complex images
        assert mse < MSE_THRESHOLD, f"MSE too high for {test_name} on GPU: {mse:.3f}"
        assert mae < MAE_THRESHOLD, f"MAE too high for {test_name} on GPU: {mae:.3f}"

        print(f"✓ GPU {test_name}: MSE={mse:.3f}, MAE={mae:.3f}")


def test_clahe_opencv_comparison_cpu():
    """Test CLAHE CPU implementation against OpenCV with MSE/MAE assertions."""
    test_images = create_synthetic_test_images()

    for test_name, test_image in test_images.items():
        # Apply OpenCV CLAHE
        opencv_result = apply_opencv_clahe(test_image, tiles_x=4, tiles_y=4, clip_limit=2.0)

        # Apply DALI CLAHE CPU
        dali_result = apply_dali_clahe_from_memory(
            test_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="cpu"
        )

        assert dali_result is not None, f"DALI CPU CLAHE failed for {test_name}"

        # Calculate metrics
        opencv_float = opencv_result.astype(np.float64)
        dali_float = dali_result.astype(np.float64)

        mse = np.mean((opencv_float - dali_float) ** 2)
        mae = np.mean(np.abs(opencv_float - dali_float))

        # Assert MSE and MAE are under 3.0
        assert mse < MSE_THRESHOLD, f"MSE too high for {test_name} on CPU: {mse:.3f}"
        assert mae < MAE_THRESHOLD, f"MAE too high for {test_name} on CPU: {mae:.3f}"

        print(f"✓ CPU {test_name}: MSE={mse:.3f}, MAE={mae:.3f}")


def test_clahe_gpu_cpu_consistency():
    """Test consistency between GPU and CPU CLAHE implementations."""
    test_images = create_synthetic_test_images()

    for test_name, test_image in test_images.items():
        # Apply DALI CLAHE on both GPU and CPU
        dali_gpu_result = apply_dali_clahe_from_memory(
            test_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="gpu"
        )
        dali_cpu_result = apply_dali_clahe_from_memory(
            test_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="cpu"
        )

        assert dali_gpu_result is not None, f"DALI GPU CLAHE failed for {test_name}"
        assert dali_cpu_result is not None, f"DALI CPU CLAHE failed for {test_name}"

        # Calculate metrics between GPU and CPU
        gpu_float = dali_gpu_result.astype(np.float64)
        cpu_float = dali_cpu_result.astype(np.float64)

        mse = np.mean((gpu_float - cpu_float) ** 2)
        mae = np.mean(np.abs(gpu_float - cpu_float))

        # GPU and CPU should be reasonably close (allow for LAB conversion differences)
        assert mse < MSE_THRESHOLD, f"MSE too high between GPU/CPU for {test_name}: {mse:.3f}"
        assert mae < MAE_THRESHOLD, f"MAE too high between GPU/CPU for {test_name}: {mae:.3f}"

        print(f"✓ GPU/CPU consistency {test_name}: MSE={mse:.3f}, MAE={mae:.3f}")


def test_clahe_different_parameters_accuracy():
    """Test CLAHE accuracy with different parameter configurations."""
    test_image = create_synthetic_test_images()["low_contrast_gradient"]

    # Test different parameter combinations
    test_configs = [
        {"tiles_x": 8, "tiles_y": 8, "clip_limit": 3.0},
        {"tiles_x": 5, "tiles_y": 7, "clip_limit": 1.0},
        {"tiles_x": 3, "tiles_y": 6, "clip_limit": 1.5},
        {"tiles_x": 4, "tiles_y": 8, "clip_limit": 2.5},
        {"tiles_x": 4, "tiles_y": 4, "clip_limit": 1.5},
    ]

    for config in test_configs:
        # Apply OpenCV CLAHE
        opencv_result = apply_opencv_clahe(test_image, **config)

        # Apply DALI CLAHE GPU and CPU
        dali_gpu_result = apply_dali_clahe_from_memory(test_image, device="gpu", **config)
        dali_cpu_result = apply_dali_clahe_from_memory(test_image, device="cpu", **config)

        assert dali_gpu_result is not None, f"DALI GPU CLAHE failed for config {config}"
        assert dali_cpu_result is not None, f"DALI CPU CLAHE failed for config {config}"

        # Calculate metrics for GPU
        opencv_float = opencv_result.astype(np.float64)
        dali_gpu_float = dali_gpu_result.astype(np.float64)
        mse_gpu = np.mean((opencv_float - dali_gpu_float) ** 2)
        mae_gpu = np.mean(np.abs(opencv_float - dali_gpu_float))

        # Calculate metrics for CPU
        dali_cpu_float = dali_cpu_result.astype(np.float64)
        mse_cpu = np.mean((opencv_float - dali_cpu_float) ** 2)
        mae_cpu = np.mean(np.abs(opencv_float - dali_cpu_float))

        # Assert accuracy for both GPU and CPU
        assert mse_gpu < MSE_THRESHOLD, f"GPU MSE too high for {config}: {mse_gpu:.3f}"
        assert mae_gpu < MAE_THRESHOLD, f"GPU MAE too high for {config}: {mae_gpu:.3f}"
        assert mse_cpu < MSE_THRESHOLD, f"CPU MSE too high for {config}: {mse_cpu:.3f}"
        assert mae_cpu < MAE_THRESHOLD, f"CPU MAE too high for {config}: {mae_cpu:.3f}"

        print(
            f"✓ Config {config}: GPU MSE={mse_gpu:.3f}, "
            "MAE={mae_gpu:.3f}; CPU MSE={mse_cpu:.3f}, MAE={mae_cpu:.3f}"
        )


def test_clahe_medical_image_accuracy():
    """Test CLAHE specifically on medical-style grayscale images."""
    medical_image = create_synthetic_test_images()["medical_scan"]

    # Apply OpenCV CLAHE
    opencv_result = apply_opencv_clahe(medical_image, tiles_x=4, tiles_y=4, clip_limit=2.0)

    # Apply DALI CLAHE on both GPU and CPU
    dali_gpu_result = apply_dali_clahe_from_memory(
        medical_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="gpu"
    )
    dali_cpu_result = apply_dali_clahe_from_memory(
        medical_image, tiles_x=4, tiles_y=4, clip_limit=2.0, device="cpu"
    )

    assert dali_gpu_result is not None, "DALI GPU CLAHE failed for medical image"
    assert dali_cpu_result is not None, "DALI CPU CLAHE failed for medical image"

    # Calculate metrics
    opencv_float = opencv_result.astype(np.float64)
    dali_gpu_float = dali_gpu_result.astype(np.float64)
    dali_cpu_float = dali_cpu_result.astype(np.float64)

    mse_gpu = np.mean((opencv_float - dali_gpu_float) ** 2)
    mae_gpu = np.mean(np.abs(opencv_float - dali_gpu_float))
    mse_cpu = np.mean((opencv_float - dali_cpu_float) ** 2)
    mae_cpu = np.mean(np.abs(opencv_float - dali_cpu_float))

    # Medical images should have very good accuracy
    assert mse_gpu < MSE_THRESHOLD, f"GPU MSE too high for medical image: {mse_gpu:.3f}"
    assert mae_gpu < MAE_THRESHOLD, f"GPU MAE too high for medical image: {mae_gpu:.3f}"
    assert mse_cpu < MSE_THRESHOLD, f"CPU MSE too high for medical image: {mse_cpu:.3f}"
    assert mae_cpu < MAE_THRESHOLD, f"CPU MAE too high for medical image: {mae_cpu:.3f}"

    print(
        f"✓ Medical image: GPU MSE={mse_gpu:.3f}, "
        "MAE={mae_gpu:.3f}; CPU MSE={mse_cpu:.3f}, MAE={mae_cpu:.3f}"
    )
