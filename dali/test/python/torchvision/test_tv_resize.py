# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import os
from typing import Sequence, Union

import numpy as np
from nose2.tools import params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

from nvidia.dali.experimental.torchvision import Resize, Compose
from nvidia.dali.backend import TensorGPU, TensorListCPU, TensorListGPU
import nvidia.dali.types as types


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
test_input_filenames = [read_filepath(fname) for fname in test_files]

to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.BOOL: torch.bool,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64,
}


def to_torch_tensor(tensor_or_tl, device_id=0):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `tensor_or_tl` : TensorGPU or TensorListGPU
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    if isinstance(dali_tensor, (TensorGPU)):
        torch_device = torch.device("cuda", device_id)
    else:
        torch_device = torch.device("cpu")
        non_blocking = False
        cuda_stream = None

    out_torch = torch.empty(
        dali_tensor.shape(),
        dtype=to_torch_type[dali_tensor.dtype],
        device=torch_device,
    )

    c_type_pointer = ctypes.c_void_p(out_torch.data_ptr())
    if isinstance(dali_tensor, (TensorGPU)):
        non_blocking = True
        cuda_stream = torch.cuda.current_stream(device=torch_device)
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor_or_tl.copy_to_external(c_type_pointer, stream, non_blocking)
    else:
        tensor_or_tl.copy_to_external(c_type_pointer)

    return out_torch


def build_resize_transform(
    resize: Union[int, Sequence[int]],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
):
    t = transforms.Compose(
        [
            transforms.Resize(
                size=resize, max_size=max_size, interpolation=interpolation, antialias=antialias
            ),
        ]
    )
    td = Compose(
        [
            Resize(
                size=resize, max_size=max_size, interpolation=interpolation, antialias=antialias
            ),
        ]
    )
    return t, td


def loop_images_test(t, td):
    for fn in test_files:
        img = Image.open(fn)

        out_tv = transforms.functional.pil_to_tensor(t(img)).unsqueeze(0).permute(0, 2, 3, 1)
        out_dali_tv = transforms.functional.pil_to_tensor(td(img)).unsqueeze(0).permute(0, 2, 3, 1)
        tv_shape_lower = torch.Size([out_tv.shape[1] - 1, out_tv.shape[2] - 1])
        tv_shape_upper = torch.Size([out_tv.shape[1] + 1, out_tv.shape[2] + 1])
        assert (
            tv_shape_lower[0] <= out_dali_tv.shape[1] <= tv_shape_upper[0]
        ), f"Should be:{out_tv.shape} is:{out_dali_tv.shape}"
        assert (
            tv_shape_lower[1] <= out_dali_tv.shape[2] <= tv_shape_upper[1]
        ), f"Should be:{out_tv.shape} is:{out_dali_tv.shape}"
        # assert torch.equal(out_tv, out_dali_tv)


@params(512, 2048, ([512, 512]), ([2048, 2048]))
def test_resize_sizes(resize):
    # Resize with single int (preserve aspect ratio)
    t, td = build_resize_transform(resize)

    loop_images_test(t, td)


@params((480, 512), (100, 124), (None, 512), (1024, 512), ([256, 256], 512))
def test_resize_max_sizes(resize, max_size):
    # Resize with single int (preserve aspect ratio)
    if resize is not None and (
        (isinstance(resize, int) and max_size is not None and max_size < resize)
        or (not isinstance(resize, int))
    ):
        with assert_raises(ValueError):
            td = Compose(
                [
                    Resize(resize, max_size=max_size),
                ]
            )
        return
    else:
        td = Compose(
            [
                Resize(resize, max_size=max_size),
            ]
        )
    t = transforms.Compose(
        [
            transforms.Resize(resize, max_size=max_size),
        ]
    )
    loop_images_test(t, td)


@params(
    ([512, 512], transforms.InterpolationMode.NEAREST),
    (1024, transforms.InterpolationMode.NEAREST_EXACT),
    ([256, 256], transforms.InterpolationMode.BILINEAR),
    (640, transforms.InterpolationMode.BICUBIC),
)
def test_resize_interploation(resize, interpolation):
    t, td = build_resize_transform(resize, interpolation=interpolation)
    loop_images_test(t, td)


@params((512, True), (2048, True), ([512, 512], True), ([2048, 2048], True))
def test_resize_antialiasing(resize, antialiasing):
    t, td = build_resize_transform(resize, antialias=antialiasing)
    loop_images_test(t, td)
