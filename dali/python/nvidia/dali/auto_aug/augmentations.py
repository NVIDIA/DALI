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

import numpy as np

from nvidia.dali import fn
from nvidia.dali.auto_aug.core import augmentation
"""
This module contains a standard suite of augmentations used by AutoAugment policy for ImageNet,
RandAugment and TrivialAugmentWide. The augmentations are implemented in terms of DALI operators.
The automatic augmentation schemes parametrize the operations with a magnitude, which,
intuitively, states how strong the given operation should be.

Each operation defines a range of magnitudes it can accept. The range is divided into a number
of bins. Then, which bin should be used for a given operation and sample is defined differently
by different automatic augmentations.

For TrivialAugment, the magnitude bins are chosen randomly every time for every sample.
In case of RandAugment, the magnitude bin is a fixed hyper-parameter.
For AutoAugment, the magnitude is a fixed parameter of a given sub-policy.

For some operations, it makes sense to randomly negate the magnitudes to increase
the variability in the augmented data. Take `fn.random` as an example. Here, the magnitude is
an angle of the rotation. Negating the angle switches if the rotation is done
clock- or counterclockwise.

The `@augmentation` lets you to specify what should the range of magnitudes and if the magnitude
should be randomly negated. Additionally, the `as_param` helps to separate the computation of
the parameter based on the magnitude from applying the operation - all the parameters will be
computed once and reused between iterations as DALI `types.Constant`.

The augmentations in this module are defined with some default ranges passed to `@augmentation`.
The parameters can be easily adjusted. For example, to increase the magnitudes range
of `shear_x`, you can create `my_shear_x = shear_x.augmentation(mag_range=(0, 0.5))`.
"""


def warp_x_param(magnitude):
    return [magnitude, 0]


def warp_y_param(magnitude):
    return [0, magnitude]


@augmentation(mag_range=(0, 0.3), randomly_negate=True, as_param=warp_x_param)
def shear_x(samples, parameter, fill_value=None, interp_type=None):
    mt = fn.transforms.shear(shear=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 0.3), randomly_negate=True, as_param=warp_y_param)
def shear_y(samples, parameter, fill_value=None, interp_type=None):
    mt = fn.transforms.shear(shear=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 0.45), randomly_negate=True, as_param=warp_x_param)
def translate_x(samples, parameter, shapes, fill_value=None, interp_type=None):
    parameter *= shapes[-2]
    mt = fn.transforms.translation(offset=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 250), randomly_negate=True, as_param=warp_x_param)
def translate_x_no_shape(samples, parameter, fill_value=None, interp_type=None):
    mt = fn.transforms.translation(offset=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 0.45), randomly_negate=True, as_param=warp_y_param)
def translate_y(samples, parameter, shapes, fill_value=None, interp_type=None):
    parameter *= shapes[-3]
    mt = fn.transforms.translation(offset=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 250), randomly_negate=True, as_param=warp_y_param)
def translate_y_no_shape(samples, parameter, fill_value=None, interp_type=None):
    mt = fn.transforms.translation(offset=parameter)
    return fn.warp_affine(samples, matrix=mt, fill_value=fill_value, interp_type=interp_type,
                          inverse_map=False)


@augmentation(mag_range=(0, 30), randomly_negate=True)
def rotate(samples, parameter, fill_value=None, interp_type=None):
    return fn.rotate(samples, angle=parameter, fill_value=fill_value, interp_type=interp_type)


def shift_enhance_range(magnitude):
    """The `enhance` operations (brightness, contrast, color, sharpness) accept magnitudes
    from [0, 2] range. However, the neutral magnitude is not 0 but 1 and the intuitive strength
    of the operation increases the further the magnitude is from 1. So, we specify magnitudes range
    to be in [0, 1] range, expect it to be randomly negated and then shift it by 1"""
    return 1 + magnitude


@augmentation(mag_range=(0, 0.9), randomly_negate=True, as_param=shift_enhance_range)
def brightness(samples, parameter):
    return fn.brightness(samples, brightness=parameter)


@augmentation(mag_range=(0, 0.9), randomly_negate=True, as_param=shift_enhance_range)
def contrast(samples, parameter):
    return fn.contrast(samples, contrast=parameter)


@augmentation(mag_range=(0, 0.9), randomly_negate=True, as_param=shift_enhance_range)
def color(samples, parameter):
    return fn.saturation(samples, saturation=parameter)


def sharpness_kernel(magnitude):
    # assumes magnitude: [-1, 1]
    blur = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13
    ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    return -magnitude * blur + (1 + magnitude) * ident


def sharpness_kernel_shifted(magnitude):
    # assumes magnitude: [0, 2]
    return sharpness_kernel(magnitude - 1)


@augmentation(mag_range=(0, 0.9), randomly_negate=True, as_param=sharpness_kernel,
              param_device="gpu")
def sharpness(samples, kernel):
    return fn.experimental.filter(samples, kernel)


def poster_mask_uint8(magnitude):
    # expects [0..8] where 8 yields identity mask and a 0
    # would be a mask that zeros all bits
    magnitude = np.round(magnitude).astype(np.uint32)
    if magnitude <= 0:
        magnitude = 1
    elif magnitude > 8:
        magnitude = 8
    nbits = np.round(8 - magnitude).astype(np.uint32)
    removal_mask = np.uint8(2)**nbits - 1
    return np.array(np.uint8(255) ^ removal_mask, dtype=np.uint8)


@augmentation(mag_range=(0, 4), as_param=poster_mask_uint8, param_device="gpu")
def posterize(samples, mask):
    return samples & mask


@augmentation(mag_range=(256, 0), param_device="gpu")
def solarize(samples, threshold):
    samples_inv = 255 - samples
    mask_unchanged = samples < threshold
    mask_inverted = 1 - mask_unchanged
    return fn.cast_like(mask_unchanged * samples + mask_inverted * samples_inv, samples)


@augmentation(mag_range=(0, 110), param_device="gpu")
def solarize_add(samples, shift, solarize_add_threshold=128):
    samples_shifted = fn.cast_like(samples + shift, samples)
    mask_shifted = samples < solarize_add_threshold
    mask_id = 1 - mask_shifted
    return fn.cast_like(mask_shifted * samples_shifted + mask_id * samples, samples)


@augmentation
def invert(samples, _):
    return fn.cast_like(255 - samples, samples)


@augmentation
def equalize(samples, _):
    return fn.experimental.equalize(samples)


@augmentation
def auto_contrast(samples, _):
    # assumes HWC layout
    lo, hi = fn.reductions.min(samples, axes=[0, 1]), fn.reductions.max(samples, axes=[0, 1])
    diff = hi - lo
    mask_scale = diff > 0
    mask_id = 1 - mask_scale
    # choose div so that scale ends up being 255 / (hi - lo) if hi > 0 and 1 otherwise
    div_by = diff * mask_scale + 255 * mask_id
    scale = 255 / div_by
    lo_scale = scale * mask_scale
    scaled = samples * scale - lo * lo_scale
    return fn.cast_like(scaled, samples)


@augmentation
def identity(samples, _):
    return samples
