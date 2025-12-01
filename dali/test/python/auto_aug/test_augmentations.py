# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from nose2.tools import params, cartesian_params

from nvidia.dali import fn, pipeline_def
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core._utils import get_translations as _get_translations

from test_utils import get_dali_extra_path, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")
vid_file = os.path.join(data_root, "db", "video", "sintel", "sintel_trailer-720p.mp4")


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


def compare_against_baseline(
    dali_aug,
    baseline_op,
    get_data,
    batch_size=default_batch_size,
    dev="gpu",
    eps=1e-7,
    max_allowed_error=1e-6,
    params=None,
    post_proc=None,
    use_shape=False,
    modality="image",
):
    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, seed=42)
    def pipeline():
        data = get_data()
        op_data = data if dev != "gpu" else data.gpu()
        mag_bin = fn.external_source(
            lambda info: np.array(info.idx_in_batch, dtype=np.int32), batch=False
        )
        extra = {}
        if use_shape:
            shape = data.shape()
            extra["shape"] = shape[int(modality == "video") :]
        output = dali_aug(op_data, num_magnitude_bins=batch_size, magnitude_bin=mag_bin, **extra)
        return output, data

    p = pipeline()
    (
        output,
        data,
    ) = p.run()
    if dev == "gpu":
        output = output.as_cpu()
    output = [np.array(sample) for sample in output]
    data = data.as_cpu()
    data = [np.array(sample) for sample in data]

    if modality == "image":

        def apply_to_sample(f, sample, *params):
            return f(sample, *params)

    else:

        def apply_to_sample(f, vid, *params):
            return np.stack([f(frame, *params) for frame in vid])

    if params is None:
        ref_output = [apply_to_sample(baseline_op, sample) for sample in data]
    else:
        assert len(params) == len(data)
        ref_output = [
            apply_to_sample(baseline_op, sample, param) for sample, param in zip(data, params)
        ]

    if post_proc is not None:
        output = [apply_to_sample(post_proc, sample) for sample in output]
        ref_output = [apply_to_sample(post_proc, sample) for sample in ref_output]
    check_batch(output, ref_output, eps=eps, max_allowed_error=max_allowed_error)


def get_images():
    image, _ = fn.readers.file(name="Reader", file_root=images_dir)
    return fn.decoders.image(image, device="cpu")


def get_videos():
    batch_size = 64
    num_vids = 4
    step = batch_size // num_vids

    def get_size(sample_info):
        size = (sample_info.idx_in_batch // step) * 25 + 200
        return np.array([size, size + 7], dtype=np.float32)

    @pipeline_def(batch_size=64, num_threads=4, device_id=0)
    def pipeline():
        image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        size = fn.external_source(source=get_size, batch=False)
        image = fn.decoders.image(image, device="cpu")
        return fn.resize(image, size=size)

    p = pipeline()
    (out,) = p.run()

    out = [np.array(sample) for sample in out]
    vids = [np.stack([out[i * step + j] for j in range(step)]) for i in range(num_vids)]

    def inner():
        return fn.external_source(
            source=lambda source_info: vids[source_info.idx_in_batch % len(vids)],
            batch=False,
            layout="FHWC",
        )

    return inner


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_shear_x(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def shear_x_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, -magnitude, 0, 0, 1, 0),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    shear_x = a.shear_x.augmentation(mag_range=(-0.3, 0.3), randomly_negate=False)
    magnitudes = shear_x._get_magnitudes(default_batch_size)
    compare_against_baseline(
        shear_x,
        pil_baseline(shear_x_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_shear_y(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def shear_y_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, -magnitude, 1, 0),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    shear_y = a.shear_y.augmentation(mag_range=(-0.3, 0.3), randomly_negate=False)
    magnitudes = shear_y._get_magnitudes(default_batch_size)
    compare_against_baseline(
        shear_y,
        pil_baseline(shear_y_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_translate_x_no_shape(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_x_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, -magnitude, 0, 1, 0),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    translate_x_no_shape = a.translate_x_no_shape.augmentation(
        mag_range=(-250, 250), randomly_negate=False
    )
    magnitudes = translate_x_no_shape._get_magnitudes(default_batch_size)
    compare_against_baseline(
        translate_x_no_shape,
        pil_baseline(translate_x_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_translate_x(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_x_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, -magnitude * img.width, 0, 1, 0),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    translate_x = a.translate_x.augmentation(mag_range=(-1, 1), randomly_negate=False)
    magnitudes = translate_x._get_magnitudes(default_batch_size)
    compare_against_baseline(
        translate_x,
        pil_baseline(translate_x_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        use_shape=True,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_translate_y_no_shape(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_y_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, -magnitude),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    translate_y_no_shape = a.translate_y_no_shape.augmentation(
        mag_range=(-250, 250), randomly_negate=False
    )
    magnitudes = translate_y_no_shape._get_magnitudes(default_batch_size)
    compare_against_baseline(
        translate_y_no_shape,
        pil_baseline(translate_y_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_translate_y(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def translate_y_ref(img, magnitude):
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, -magnitude * img.height),
            Image.BILINEAR,
            fillcolor=(128,) * 3,
        )

    data_source = get_images if modality == "image" else get_videos()
    translate_y = a.translate_y.augmentation(mag_range=(-1, 1), randomly_negate=False)
    magnitudes = translate_y._get_magnitudes(default_batch_size)
    compare_against_baseline(
        translate_y,
        pil_baseline(translate_y_ref),
        data_source,
        dev=dev,
        params=magnitudes,
        use_shape=True,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_rotate(modality, dev):
    # adapted implementation from DeepLearningExamples:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    # Classification/ConvNets/image_classification/autoaugment.py
    def rotate_with_fill(img, magnitude):
        rot = img.convert("RGBA").rotate(magnitude, resample=Image.BILINEAR)
        return Image.composite(rot, Image.new("RGBA", img.size, (128,) * 3), rot).convert(img.mode)

    data_source = get_images if modality == "image" else get_videos()
    rotate = a.rotate.augmentation(mag_range=(-30, 30), randomly_negate=False)
    magnitudes = rotate._get_magnitudes(default_batch_size)
    compare_against_baseline(
        rotate,
        pil_baseline(rotate_with_fill),
        data_source,
        dev=dev,
        params=magnitudes,
        max_allowed_error=None,
        eps=1,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_brightness(modality, dev):
    def brightness_ref(img, magnitude):
        return ImageEnhance.Brightness(img).enhance(magnitude)

    data_source = get_images if modality == "image" else get_videos()
    brightness = a.brightness.augmentation(
        mag_range=(0.1, 1.9), randomly_negate=False, mag_to_param=None
    )
    magnitudes = brightness._get_magnitudes(default_batch_size)
    compare_against_baseline(
        brightness,
        pil_baseline(brightness_ref),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=magnitudes,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_contrast(modality, dev):
    def contrast_ref(img, magnitude):
        return ImageEnhance.Contrast(img).enhance(magnitude)

    data_source = get_images if modality == "image" else get_videos()
    contrast = a.contrast.augmentation(
        mag_range=(0.1, 1.9), randomly_negate=False, mag_to_param=None
    )
    magnitudes = contrast._get_magnitudes(default_batch_size)
    compare_against_baseline(
        contrast,
        pil_baseline(contrast_ref),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=magnitudes,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_color(modality, dev):
    max_allowed_error = 2

    def color_ref(img, magnitude):
        return ImageEnhance.Color(img).enhance(magnitude)

    data_source = get_images if modality == "image" else get_videos()
    color = a.color.augmentation(mag_range=(0.1, 1.9), randomly_negate=False, mag_to_param=None)
    magnitudes = color._get_magnitudes(default_batch_size)
    compare_against_baseline(
        color,
        pil_baseline(color_ref),
        data_source,
        max_allowed_error=max_allowed_error,
        dev=dev,
        params=magnitudes,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_sharpness(modality, dev):
    def sharpness_ref(img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(magnitude)

    def post_proc(img):
        # pill applies convolution in valid mode (so for 3x3 kernel used,
        # the output is smaller by one pixel at each end, and then pastes
        # the filtered image onto the original). We don't do the
        # pasting
        return img[1:-1, 1:-1, :]

    data_source = get_images if modality == "image" else get_videos()
    sharpness = a.sharpness.augmentation(
        mag_range=(0.1, 1.9), randomly_negate=False, mag_to_param=a.sharpness_kernel_shifted
    )
    magnitudes = sharpness._get_magnitudes(default_batch_size)
    compare_against_baseline(
        sharpness,
        pil_baseline(sharpness_ref),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=magnitudes,
        post_proc=post_proc,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_posterize(modality, dev):
    data_source = get_images if modality == "image" else get_videos()
    # note, 0 is remapped to 1 as in tf implementation referred in the RA paper, thus (1, 8) range
    posterize = a.posterize.augmentation(param_device=dev, mag_range=(1, 8))
    magnitudes = np.round(posterize._get_magnitudes(default_batch_size)).astype(np.int32)
    compare_against_baseline(
        posterize,
        pil_baseline(ImageOps.posterize),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=magnitudes,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_solarize(modality, dev):
    data_source = get_images if modality == "image" else get_videos()
    solarize = a.solarize.augmentation(param_device=dev)
    magnitudes = solarize._get_magnitudes(default_batch_size)
    params = solarize._map_mags_to_params(magnitudes)
    compare_against_baseline(
        solarize,
        pil_baseline(ImageOps.solarize),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=params,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_solarize_add(modality, dev):
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

    data_source = get_images if modality == "image" else get_videos()
    solarize_add = a.solarize_add.augmentation(param_device=dev)
    magnitudes = solarize_add._get_magnitudes(default_batch_size)
    params = solarize_add._map_mags_to_params(magnitudes)
    compare_against_baseline(
        solarize_add,
        pil_baseline(solarize_add_ref),
        data_source,
        max_allowed_error=1,
        dev=dev,
        params=params,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_invert(modality, dev):
    data_source = get_images if modality == "image" else get_videos()
    compare_against_baseline(
        a.invert,
        pil_baseline(ImageOps.invert),
        data_source,
        max_allowed_error=1,
        dev=dev,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_equalize(modality, dev):
    # pil's equalization uses slightly different formula when
    # transforming cumulative-sum of histogram into lookup table than open-cv
    # so the point-wise diffs can be significant, but the average is not
    data_source = get_images if modality == "image" else get_videos()
    compare_against_baseline(
        a.equalize,
        pil_baseline(ImageOps.equalize),
        data_source,
        max_allowed_error=None,
        dev=dev,
        eps=7,
        modality=modality,
    )


@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_auto_contrast(modality, dev):
    data_source = get_images if modality == "image" else get_videos()
    compare_against_baseline(
        a.auto_contrast,
        pil_baseline(ImageOps.autocontrast),
        data_source,
        max_allowed_error=1,
        dev=dev,
        modality=modality,
    )


# Check edge cases (single-value channels)
@cartesian_params(("image", "video"), ("cpu", "gpu"))
def test_auto_contrast_mono_channels(modality, dev):
    def modal_shape(shape, num_frames=15):
        if modality != "video":
            return shape
        return (num_frames,) + shape

    rng = np.random.default_rng(seed=42)
    const_single_channel = np.full(modal_shape((101, 205, 1)), 0, dtype=np.uint8)
    const_multi_channel = np.full(modal_shape((200, 512, 3)), 255, dtype=np.uint8)
    const_multi_per_channel = np.stack(
        [
            np.full(modal_shape((300, 300), 7), 254, dtype=np.uint8),
            np.full(modal_shape((300, 300), 7), 159, dtype=np.uint8),
            np.full(modal_shape((300, 300), 7), 1, dtype=np.uint8),
        ],
        axis=-1,
    )
    rnd_uniform_and_fixed_channel = np.stack(
        [
            np.uint8(rng.uniform(50, 160, modal_shape((400, 400), 32))),
            np.full(modal_shape((400, 400), 32), 159, dtype=np.uint8),
            np.uint8(rng.uniform(0, 255, modal_shape((400, 400), 32))),
        ],
        axis=-1,
    )
    imgs = [
        const_single_channel,
        const_multi_channel,
        const_multi_per_channel,
        rnd_uniform_and_fixed_channel,
    ]

    def get_batch():
        layout = "HWC" if modality != "video" else "FHWC"
        return fn.external_source(lambda: imgs, batch=True, layout=layout)

    compare_against_baseline(
        a.auto_contrast,
        pil_baseline(ImageOps.autocontrast),
        get_batch,
        batch_size=len(imgs),
        max_allowed_error=1,
        dev=dev,
        modality=modality,
    )


@params(*tuple(itertools.product((True, False), (0, 1), ("height", "width", "both", "none"))))
def test_translation_helper(use_shape, offset_fraction, extent):
    # make sure the translation helper processes the args properly
    # note, it only uses translate_y (as it is in imagenet policy)
    default_abs = 123
    default_rel = 0.123
    height, width = 300, 700
    shape = [height, width]
    params = {}
    assert extent in ("height", "width", "both", "none"), f"{extent}"
    if extent != "none":
        if use_shape:
            param_shape = [1.0, 1.0]
            param_name = "max_translate_rel"
        else:
            param_shape = shape
            param_name = "max_translate_abs"
        if extent == "both":
            param = [param_shape[0] * offset_fraction, param_shape[1] * offset_fraction]
        elif extent == "height":
            param = [param_shape[0] * offset_fraction, 0]
        else:
            assert extent == "width"
            param = [0, param_shape[1] * offset_fraction]
        params[param_name] = param

    translate_x, translate_y = _get_translations(use_shape, default_abs, default_rel, **params)

    if use_shape:
        assert translate_x.op is a.translate_x.op
        assert translate_y.op is a.translate_y.op
    else:
        assert translate_x.op is a.translate_x_no_shape.op
        assert translate_y.op is a.translate_y_no_shape.op

    mag_ranges = [translate_x.mag_range, translate_y.mag_range]

    if extent == "none":
        expected_height = default_rel if use_shape else default_abs
        expected_width = expected_height
    elif use_shape:
        expected_height = offset_fraction
        expected_width = offset_fraction
    else:
        expected_height = height * offset_fraction
        expected_width = width * offset_fraction

    if extent == "both":
        assert translate_x.mag_range == (0, expected_width), f"{mag_ranges} {expected_width}"
        assert translate_y.mag_range == (0, expected_height), f"{mag_ranges} {expected_height}"
    elif extent == "height":
        assert translate_x.mag_range == (0, 0), f"{mag_ranges}"
        assert translate_y.mag_range == (0, expected_height), f"{mag_ranges} {expected_height}"
    elif extent == "width":
        assert translate_x.mag_range == (0, expected_width), f"{mag_ranges} {expected_width}"
        assert translate_y.mag_range == (0, 0), f"{mag_ranges}"
    else:
        assert extent == "none"
        assert translate_x.mag_range == (0, expected_width), f"{mag_ranges}"
        assert translate_y.mag_range == (0, expected_height), f"{mag_ranges}"
