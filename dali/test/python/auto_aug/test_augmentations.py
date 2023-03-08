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

import os

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from nose2.tools import params

import nvidia.dali.tensors as _tensors
from nvidia.dali import pipeline_def, fn
from nvidia.dali.auto_aug import augmentations as a

from test_utils import get_dali_extra_path, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


def maybe_squeeze(img, axis=2):
    if img.shape[axis] != 1:
        return img
    return np.squeeze(img, axis=axis)


def maybe_expand(img, axis=2):
    if len(img.shape) != axis:
        return img
    return np.expand_dims(img, axis=axis)


def pil_baseline(pil_op):

    def inner(sample, param=None):
        p_sample = Image.fromarray(maybe_squeeze(sample))
        p_out = pil_op(p_sample) if param is None else pil_op(p_sample, param)
        return maybe_expand(np.array(p_out))

    return inner


def compare_against_baseline(dali_aug, baseline_op, get_data, batch_size, dev="gpu",
                             max_allowed_error=1e-6, params=None, post_proc=None):

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, seed=42)
    def pipeline():
        data = get_data()
        op_data = data if dev != "gpu" else data.gpu()
        mag_bin = fn.external_source(lambda info: np.array(info.idx_in_batch, dtype=np.int32),
                                     batch=False)
        output = dali_aug(op_data, num_magnitude_bins=batch_size, magnitude_bin=mag_bin)
        return output, data

    p = pipeline()
    p.build()
    output, data, = p.run()
    if dev == "gpu":
        output = output.as_cpu()
    output = [np.array(sample) for sample in output]
    if isinstance(data, _tensors.TensorListGPU):
        data = data.as_cpu()
    data = [np.array(sample) for sample in data]

    if params is None:
        ref_output = [baseline_op(sample) for sample in data]
    else:
        assert len(params) == len(data)
        ref_output = [baseline_op(sample, param) for sample, param in zip(data, params)]

    if post_proc is not None:
        output = [post_proc(sample) for sample in output]
        ref_output = [post_proc(sample) for sample in ref_output]
    check_batch(output, ref_output, max_allowed_error=max_allowed_error)


def get_images(dev):

    def inner():
        image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        return fn.decoders.image(image, device="cpu" if dev == "cpu" else "mixed")

    return inner


@params(("gpu", ))
def test_sharpness(dev):

    def sharpness_ref(img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(magnitude)

    def post_proc(img):
        # pill applies convolution in valid mode (so for 3x3 kernel used,
        # the output is smaller by one pixel at each end, and then pastes
        # the filtered image onto the original). We don't do the
        # pasting
        return img[1:-1, 1:-1, :]

    batch_size = 16
    data_source = get_images(dev)
    sharpness = a.sharpness.augmentation(mag_range=(0.1, 1.9), randomly_negate=False,
                                         as_param=a.sharpness_kernel_shifted)
    magnitudes = sharpness._get_magnitudes(batch_size)
    compare_against_baseline(sharpness, pil_baseline(sharpness_ref), data_source,
                             batch_size=batch_size, max_allowed_error=1, dev=dev, params=magnitudes,
                             post_proc=post_proc)


@params(("cpu", ), ("gpu", ))
def test_posterize(dev):
    batch_size = 16
    data_source = get_images(dev)
    # note, 0 is remapped to 1 as in tf implementation referred in the RA paper, thus (1, 8) range
    posterize = a.posterize.augmentation(param_device=dev, mag_range=(1, 8))
    magnitudes = np.round(posterize._get_magnitudes(batch_size)).astype(np.int32)
    compare_against_baseline(posterize, pil_baseline(ImageOps.posterize), data_source,
                             batch_size=batch_size, max_allowed_error=1, dev=dev, params=magnitudes)


@params(("cpu", ), ("gpu", ))
def test_solarize(dev):
    batch_size = 16
    data_source = get_images(dev)
    solarize = a.solarize.augmentation(param_device=dev)
    magnitudes = solarize._get_magnitudes(batch_size)
    params = solarize._map_mags_to_params(magnitudes)
    compare_against_baseline(solarize, pil_baseline(ImageOps.solarize), data_source,
                             batch_size=batch_size, max_allowed_error=1, dev=dev, params=params)


@params(("cpu", ), ("gpu", ))
def test_solarize_add(dev):

    # the implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def solarize_add_ref(image, magnitude):
        threshold = 128
        lut = []
        for i in range(256):
            if i < threshold:
                res = i + magnitude if i + magnitude <= 255 else 255
                res = res if res >= 0 else 0
                lut.append(res)
            else:
                lut.append(i)
        return ImageOps._lut(image, lut)

    batch_size = 16
    data_source = get_images(dev)
    solarize_add = a.solarize_add.augmentation(param_device=dev)
    magnitudes = solarize_add._get_magnitudes(batch_size)
    params = solarize_add._map_mags_to_params(magnitudes)
    compare_against_baseline(solarize_add, pil_baseline(solarize_add_ref), data_source,
                             batch_size=batch_size, max_allowed_error=1, dev=dev, params=params)


@params(("cpu", ), ("gpu", ))
def test_invert(dev):
    data_source = get_images(dev)
    compare_against_baseline(a.invert, pil_baseline(ImageOps.invert), data_source, batch_size=16,
                             max_allowed_error=1, dev=dev)


@params(("cpu", ), ("gpu", ))
def test_auto_contrast(dev):
    data_source = get_images(dev)
    compare_against_baseline(a.auto_contrast, pil_baseline(ImageOps.autocontrast), data_source,
                             batch_size=16, max_allowed_error=1, dev=dev)


# Check edge cases (single-value channels)
@params(("cpu", ), ("gpu", ))
def test_auto_contrast_mono_channels(dev):
    rng = np.random.default_rng(seed=42)
    const_single_channel = np.full((101, 205, 1), 0, dtype=np.uint8)
    const_multi_channel = np.full((200, 512, 3), 255, dtype=np.uint8)
    const_multi_per_channel = np.stack([
        np.full((300, 300), 254, dtype=np.uint8),
        np.full((300, 300), 159, dtype=np.uint8),
        np.full((300, 300), 1, dtype=np.uint8)
    ], axis=2)
    rnd_uniform_and_fixed_channel = np.stack([
        np.uint8(rng.uniform(50, 160, (400, 400))),
        np.full((400, 400), 159, dtype=np.uint8),
        np.uint8(rng.uniform(0, 255, (400, 400))),
    ], axis=2)
    imgs = [
        const_single_channel, const_multi_channel, const_multi_per_channel,
        rnd_uniform_and_fixed_channel
    ]

    def get_batch():
        return fn.external_source(lambda: imgs, batch=True)

    compare_against_baseline(a.auto_contrast, pil_baseline(ImageOps.autocontrast), get_batch,
                             batch_size=len(imgs), max_allowed_error=1, dev=dev)
