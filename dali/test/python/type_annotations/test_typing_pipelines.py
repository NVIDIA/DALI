# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn, ops
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali import types, tensors

from test_utils import get_dali_extra_path

_test_root = Path(get_dali_extra_path())


@pipeline_def(batch_size=10, device_id=0, num_threads=4)
def rn50_pipe():
    enc, label = fn.readers.file(
        files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")], name="FileReader")
    imgs = fn.decoders.image(enc, device="mixed")
    rng = fn.random.coin_flip(probability=0.5)
    resized = fn.random_resized_crop(imgs, size=[224, 224])
    normalized = fn.crop_mirror_normalize(resized, mirror=rng,
                                          dtype=types.DALIDataType.FLOAT16,
                                          output_layout="HWC", crop=(224, 224),
                                          mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return normalized, label.gpu()


def test_rn50_pipe():
    pipe = rn50_pipe()
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.FLOAT16  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)


@pipeline_def(batch_size=10, device_id=0, num_threads=4)
def rn50_ops_pipe():
    Reader = ops.readers.File(files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")],
                              name="FileReader")
    Decoder = ops.decoders.Image(device="mixed")
    Rng = ops.random.CoinFlip(probability=0.5)
    Rrc = ops.RandomResizedCrop(device="gpu", size=[224, 224])
    Cmn = ops.CropMirrorNormalize(mirror=Rng(), device="gpu", dtype=types.DALIDataType.FLOAT16,
                                  output_layout="HWC", crop=(224, 224),
                                  mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                  std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    enc, label = Reader()
    imgs = Decoder(enc)
    resized = Rrc(imgs)
    normalized = Cmn(resized)
    return normalized, label.gpu()


def test_rn50_ops_pipe():
    pipe = rn50_ops_pipe()
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.FLOAT16  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)


@pipeline_def(batch_size=10, device_id=0, num_threads=4, enable_conditionals=True)
def cond_pipe():
    enc, label = fn.readers.file(
        files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")], name="FileReader")
    imgs = fn.decoders.image(enc, device="mixed")
    resized = fn.resize(imgs, size=[224, 224], interp_type=types.DALIInterpType.INTERP_LINEAR)
    if fn.random.uniform(range=[0, 1]) < 0.25:
        out = fn.rotate(resized, angle=fn.random.uniform(range=[30, 60]))
    else:
        out = resized
    return out, label.gpu()


def test_cond_pipe():
    pipe = cond_pipe()
    pipe.build()
    imgs, labels = pipe.run()
    assert isinstance(imgs, tensors.TensorListGPU)
    assert imgs.dtype == types.DALIDataType.UINT8  # noqa: E721
    assert isinstance(labels, tensors.TensorListGPU)

@pipeline_def(batch_size=10, device_id=0, num_threads=4)
def es_pipe():
    single_output = fn.external_source(source=lambda: np.array([0]), batch=False)
    fist_output, second_output = fn.external_source(source=lambda: (np.array([1]), np.array([2])),
                                                    num_outputs=2, batch=False)
    return single_output, fist_output, second_output


def test_es_pipe():
    pipe = es_pipe()
    pipe.build()
    out0, out1, out2 = pipe.run()
    assert np.array_equal(np.array(out0.as_tensor()), np.full((10, 1), 0))
    assert np.array_equal(np.array(out1.as_tensor()), np.full((10, 1), 1))
    assert np.array_equal(np.array(out2.as_tensor()), np.full((10, 1), 2))
