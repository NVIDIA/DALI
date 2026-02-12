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

from pathlib import Path

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose_utils import attr  # type: ignore
from nvidia.dali import types
from test_utils import get_dali_extra_path

_test_root = Path(get_dali_extra_path())


# Use annotated function for additional verification. Anything that is not the expected type
# passed here would fail mypy checks, as well as printing the runtime error
def expect_tensor(*inputs: ndd.Tensor) -> None:
    for input in inputs:
        assert isinstance(input, ndd.Tensor), f"Expected Tensor, got {input} of type {type(input)}"


def expect_batch(*inputs: ndd.Batch) -> None:
    for input in inputs:
        assert isinstance(input, ndd.Batch), f"Expected Batch, got {input} of type {type(input)}"


def test_rn50_sample():
    reader = ndd.readers.File(files=str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg"))
    for enc, label in reader.next_epoch():
        img = ndd.decoders.image(enc, device="gpu")
        rng = ndd.random.coin_flip(probability=0.5)
        resized = ndd.random_resized_crop(img, size=[224, 224])
        normalized = ndd.crop_mirror_normalize(
            resized,
            mirror=rng,
            dtype=types.DALIDataType.FLOAT16,
            output_layout="HWC",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        expect_tensor(enc, label, img, rng, resized, normalized, label.gpu())
        break


def test_rn50_batch():
    reader = ndd.readers.File(files=str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg"))
    for enc, labels in reader.next_epoch(batch_size=10):
        imgs = ndd.decoders.image(enc, device="gpu")
        rng = ndd.random.coin_flip(probability=0.5, batch_size=10)
        resized = ndd.random_resized_crop(imgs, size=[224, 224])
        normalized = ndd.crop_mirror_normalize(
            resized,
            mirror=rng,
            dtype=types.DALIDataType.FLOAT16,
            output_layout="HWC",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        expect_batch(enc, labels, imgs, rng, resized, normalized, labels.gpu())
        break


def test_broadcast():
    tensor = ndd.zeros(shape=10)
    batch = ndd.ones(shape=10, batch_size=2)

    result_tensor = ndd.stack(tensor, tensor)
    result_batch = ndd.stack(tensor, batch)

    expect_tensor(tensor, result_tensor)
    expect_batch(batch, result_batch)


@attr("pytorch")
def test_copy_tensor_constant():
    import torch  # type: ignore

    const_int = ndd.copy(1)
    const_float = ndd.copy(2.0)
    const_list = ndd.copy([2, 3])
    const_tuple = ndd.copy((4, 5))
    const_torch = ndd.copy(torch.full((2, 2), 6))
    const_np = ndd.copy(np.full((2, 2), 7))
    expect_tensor(const_int, const_float, const_list, const_tuple, const_torch, const_np)

    assert np.array_equal(const_int, 1)
    assert np.array_equal(const_float, 2.0)
    assert np.array_equal(const_list, [2, 3])
    assert np.array_equal(const_tuple, [4, 5])
    assert np.array_equal(const_torch, np.full((2, 2), 6))
    assert np.array_equal(const_np, np.full((2, 2), 7))
