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


default_batch_size = 16


def compare_against_baseline(dali_aug, baseline_op, get_data, batch_size=default_batch_size,
                             dev="gpu", eps=1e-7, max_allowed_error=1e-6, params=None,
                             post_proc=None, use_shape=False):

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, seed=42)
    def pipeline():
        data = get_data()
        op_data = data if dev != "gpu" else data.gpu()
        mag_bin = fn.external_source(lambda info: np.array(info.idx_in_batch, dtype=np.int32),
                                     batch=False)
        extra = {} if not use_shape else {"shape": fn.shapes(data)}
        output = dali_aug(op_data, num_magnitude_bins=batch_size, magnitude_bin=mag_bin, **extra)
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
    check_batch(output, ref_output, eps=eps, max_allowed_error=max_allowed_error)


def get_images():

    def inner():
        image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        return fn.decoders.image(image, device="cpu")

    return inner


@params(("cpu", ), ("gpu", ))
def test_shear_x(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def shear_x_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, -magnitude, 0, 0, 1, 0), Image.BILINEAR,
                             fillcolor=(128, ) * 3)

    data_source = get_images()
    shear_x = a.shear_x.augmentation(mag_range=(-0.3, 0.3), randomly_negate=False)
    magnitudes = shear_x._get_magnitudes(default_batch_size)
    compare_against_baseline(shear_x, pil_baseline(shear_x_ref), data_source, dev=dev,
                             params=magnitudes, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_shear_y(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def shear_y_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, -magnitude, 1, 0), Image.BILINEAR,
                             fillcolor=(128, ) * 3)

    data_source = get_images()
    shear_y = a.shear_y.augmentation(mag_range=(-0.3, 0.3), randomly_negate=False)
    magnitudes = shear_y._get_magnitudes(default_batch_size)
    compare_against_baseline(shear_y, pil_baseline(shear_y_ref), data_source, dev=dev,
                             params=magnitudes, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_translate_x_no_shape(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_x_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, -magnitude, 0, 1, 0), Image.BILINEAR,
                             fillcolor=(128, ) * 3)

    data_source = get_images()
    translate_x_no_shape = a.translate_x_no_shape.augmentation(mag_range=(-250, 250),
                                                               randomly_negate=False)
    magnitudes = translate_x_no_shape._get_magnitudes(default_batch_size)
    compare_against_baseline(translate_x_no_shape, pil_baseline(translate_x_ref), data_source,
                             dev=dev, params=magnitudes, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_translate_x(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_x_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, -magnitude * img.width, 0, 1, 0),
                             Image.BILINEAR, fillcolor=(128, ) * 3)

    data_source = get_images()
    translate_x = a.translate_x.augmentation(mag_range=(-1, 1), randomly_negate=False)
    magnitudes = translate_x._get_magnitudes(default_batch_size)
    compare_against_baseline(translate_x, pil_baseline(translate_x_ref), data_source, dev=dev,
                             params=magnitudes, use_shape=True, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_translate_y_no_shape(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_y_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, -magnitude), Image.BILINEAR,
                             fillcolor=(128, ) * 3)

    data_source = get_images()
    translate_y_no_shape = a.translate_y_no_shape.augmentation(mag_range=(-250, 250),
                                                               randomly_negate=False)
    magnitudes = translate_y_no_shape._get_magnitudes(default_batch_size)
    compare_against_baseline(translate_y_no_shape, pil_baseline(translate_y_ref), data_source,
                             dev=dev, params=magnitudes, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_translate_y(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_y_ref(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, -magnitude * img.height),
                             Image.BILINEAR, fillcolor=(128, ) * 3)

    data_source = get_images()
    translate_y = a.translate_y.augmentation(mag_range=(-1, 1), randomly_negate=False)
    magnitudes = translate_y._get_magnitudes(default_batch_size)
    compare_against_baseline(translate_y, pil_baseline(translate_y_ref), data_source, dev=dev,
                             params=magnitudes, use_shape=True, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_rotate(dev):

    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def rotate_with_fill(img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude, resample=Image.BILINEAR)
        return Image.composite(rot, Image.new("RGBA", img.size, (128, ) * 3), rot).convert(img.mode)

    data_source = get_images()
    rotate = a.rotate.augmentation(mag_range=(-30, 30), randomly_negate=False)
    magnitudes = rotate._get_magnitudes(default_batch_size)
    compare_against_baseline(rotate, pil_baseline(rotate_with_fill), data_source, dev=dev,
                             params=magnitudes, max_allowed_error=None, eps=1)


@params(("cpu", ), ("gpu", ))
def test_brightness(dev):

    def brightness_ref(img, magnitude):
        return ImageEnhance.Brightness(img).enhance(magnitude)

    data_source = get_images()
    brightness = a.brightness.augmentation(mag_range=(0.1, 1.9), randomly_negate=False,
                                           as_param=None)
    magnitudes = brightness._get_magnitudes(default_batch_size)
    compare_against_baseline(brightness, pil_baseline(brightness_ref), data_source,
                             max_allowed_error=1, dev=dev, params=magnitudes)


@params(("cpu", ), ("gpu", ))
def test_contrast(dev):

    def contrast_ref(img, magnitude):
        return ImageEnhance.Contrast(img).enhance(magnitude)

    data_source = get_images()
    contrast = a.contrast.augmentation(mag_range=(0.1, 1.9), randomly_negate=False, as_param=None)
    magnitudes = contrast._get_magnitudes(default_batch_size)
    compare_against_baseline(contrast, pil_baseline(contrast_ref), data_source, max_allowed_error=1,
                             dev=dev, params=magnitudes)


@params(("cpu", ), ("gpu", ))
def test_color(dev):
    max_allowed_error = 2

    def color_ref(img, magnitude):
        return ImageEnhance.Color(img).enhance(magnitude)

    data_source = get_images()
    color = a.color.augmentation(mag_range=(0.1, 1.9), randomly_negate=False, as_param=None)
    magnitudes = color._get_magnitudes(default_batch_size)
    compare_against_baseline(color, pil_baseline(color_ref), data_source,
                             max_allowed_error=max_allowed_error, dev=dev, params=magnitudes)


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

    data_source = get_images()
    sharpness = a.sharpness.augmentation(mag_range=(0.1, 1.9), randomly_negate=False,
                                         as_param=a.sharpness_kernel_shifted)
    magnitudes = sharpness._get_magnitudes(default_batch_size)
    compare_against_baseline(sharpness, pil_baseline(sharpness_ref), data_source,
                             max_allowed_error=1, dev=dev, params=magnitudes, post_proc=post_proc)


@params(("cpu", ), ("gpu", ))
def test_posterize(dev):
    data_source = get_images()
    # note, 0 is remapped to 1 as in tf implementation referred in the RA paper, thus (1, 8) range
    posterize = a.posterize.augmentation(param_device=dev, mag_range=(1, 8))
    magnitudes = np.round(posterize._get_magnitudes(default_batch_size)).astype(np.int32)
    compare_against_baseline(posterize, pil_baseline(ImageOps.posterize), data_source,
                             max_allowed_error=1, dev=dev, params=magnitudes)


@params(("cpu", ), ("gpu", ))
def test_solarize(dev):
    data_source = get_images()
    solarize = a.solarize.augmentation(param_device=dev)
    magnitudes = solarize._get_magnitudes(default_batch_size)
    params = solarize._map_mags_to_params(magnitudes)
    compare_against_baseline(solarize, pil_baseline(ImageOps.solarize), data_source,
                             max_allowed_error=1, dev=dev, params=params)


@params(("cpu", ), ("gpu", ))
def test_solarize_add(dev):

    # adapted the implementation from DeepLearningExamples:
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

    data_source = get_images()
    solarize_add = a.solarize_add.augmentation(param_device=dev)
    magnitudes = solarize_add._get_magnitudes(default_batch_size)
    params = solarize_add._map_mags_to_params(magnitudes)
    compare_against_baseline(solarize_add, pil_baseline(solarize_add_ref), data_source,
                             max_allowed_error=1, dev=dev, params=params)


@params(("cpu", ), ("gpu", ))
def test_invert(dev):
    data_source = get_images()
    compare_against_baseline(a.invert, pil_baseline(ImageOps.invert), data_source,
                             max_allowed_error=1, dev=dev)


@params(("gpu", ))
def test_equalize(dev):

    # pil's equalization uses slightly different formula when
    # transforming cumulative-sum of histogram into lookup table than open-cv
    # so the point-wise diffs can be significant, but the average is not
    # (comparable to geom transforms)
    data_source = get_images()
    compare_against_baseline(a.equalize, pil_baseline(ImageOps.equalize), data_source,
                             max_allowed_error=None, dev=dev, eps=7)


@params(("cpu", ), ("gpu", ))
def test_auto_contrast(dev):
    data_source = get_images()
    compare_against_baseline(a.auto_contrast, pil_baseline(ImageOps.autocontrast), data_source,
                             max_allowed_error=1, dev=dev)


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
