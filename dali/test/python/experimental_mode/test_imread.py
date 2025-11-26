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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.backend as _backend
import nvidia.dali.types as types
from nose_utils import SkipTest
from test_utils import get_dali_extra_path
import os
import pathlib
import numpy as np
from nose2.tools import cartesian_params

dali_extra_path = get_dali_extra_path()
tiff_dir = os.path.join(dali_extra_path, "db", "single", "tiff")
reference_dir = os.path.join(dali_extra_path, "db", "single", "reference", "tiff")

# string paths
images = [
    os.path.join(tiff_dir, "0", "cat-111793_640.tiff"),
    os.path.join(tiff_dir, "0", "cat-3449999_640.tiff"),
    os.path.join(tiff_dir, "0", "cat-3504008_640.tiff"),
]

# Also provide images as pathlib.Path objects for additional tests
images_pathlib = [
    pathlib.Path(tiff_dir) / "0" / "cat-111793_640.tiff",
    pathlib.Path(tiff_dir) / "0" / "cat-3449999_640.tiff",
    pathlib.Path(tiff_dir) / "0" / "cat-3504008_640.tiff",
]

# Create Tensor/Batch with filepath bytes (not image contents)
images_tensors = [
    ndd.tensor(np.frombuffer(str(image).encode("utf-8"), dtype=np.uint8).copy()) for image in images
]

images_batch = ndd.batch(images_tensors)

references_rgb = [
    os.path.join(reference_dir, "0", "cat-111793_640.tiff.npy"),
    os.path.join(reference_dir, "0", "cat-3449999_640.tiff.npy"),
    os.path.join(reference_dir, "0", "cat-3504008_640.tiff.npy"),
]

references_gray = [
    os.path.join(reference_dir, "0", "cat-111793_640_gray.tiff.npy"),
    os.path.join(reference_dir, "0", "cat-3449999_640_gray.tiff.npy"),
    os.path.join(reference_dir, "0", "cat-3504008_640_gray.tiff.npy"),
]

references = {types.RGB: references_rgb, types.GRAY: references_gray}


@cartesian_params(
    ("cpu", "mixed"),
    ({}, {"output_type": types.GRAY}),
    (
        images[0],
        images_pathlib[0],
        images_tensors[0],
        images,
        images_pathlib,
        images_tensors,
        tuple(images),
        tuple(images_pathlib),
        tuple(images_tensors),
        images_batch,
    ),
)
def test_imread(device_type, decoder_kwargs, image_paths):
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    output_device = "cpu" if device_type == "cpu" else "gpu"

    if isinstance(image_paths, (ndd.Tensor, str, pathlib.Path)):
        out_type = ndd.Tensor
    elif isinstance(image_paths, (list, tuple, ndd.Batch)):
        out_type = ndd.Batch
    else:
        raise ValueError(
            f"Invalid image paths type: {type(image_paths)}. Must be Tensor, str, "
            f"pathlib.Path, list, tuple, or Batch."
        )

    result = ndd.imread(image_paths, device=device_type, **decoder_kwargs)
    assert isinstance(result, out_type)
    assert result.device == ndd.Device(output_device)

    output_type = decoder_kwargs.get("output_type", types.RGB)
    expected_nchannels = 1 if output_type == types.GRAY else 3

    def to_cpu_array(t):
        arr = t.cpu() if t.device.device_type == "gpu" else t
        return np.array(arr)

    tensors = [result] if out_type == ndd.Tensor else result.tensors
    reference_files = references[output_type][: len(tensors)]

    if out_type == ndd.Tensor:
        assert len(result.shape) == 3  # HWC
        assert result.shape[2] == expected_nchannels
        assert result.dtype == ndd.uint8
    else:
        assert result.batch_size == len(reference_files)
        assert result.dtype == ndd.uint8

    for tensor, reference in zip(tensors, reference_files):
        reference_array = np.load(reference)
        decoded_array = to_cpu_array(tensor)
        np.testing.assert_allclose(decoded_array, reference_array, atol=1)
