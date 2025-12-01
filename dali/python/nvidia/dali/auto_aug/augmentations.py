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

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples."
    )

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug.core import augmentation

"""
This module contains a standard suite of augmentations used by AutoAugment policy for ImageNet,
RandAugment and TrivialAugmentWide. The augmentations are implemented in terms of DALI operators.

The `@augmentation` decorator handles computation of the decorated transformations's parameter.
When called, the decorated augmentation expects:
* a single positional argument: batch of samples
* `magnitude_bin` and `num_magnitude_bins` instead of the parameter.
  The parameter is computed as if by calling
  `mag_to_param(magnitudes[magnitude_bin] * ((-1) ** random_sign))`, where
  `magnitudes=linspace(mag_range[0], mag_range[1], num_magnitude_bins)`.

The augmentations in this module are defined with example setups passed
to `@augmentation`. The parameters can be easily adjusted. For instance, to increase
the magnitudes range of `shear_x` from 0.3 to 0.5, you can create
`my_shear_x = shear_x.augmentation(mag_range=(0, 0.5))`.
"""


def warp_x_param(magnitude):
    return [magnitude, 0]


def warp_y_param(magnitude):
    return [0, magnitude]


@augmentation(mag_range=(0, 0.3), randomly_negate=True, mag_to_param=warp_x_param)
def shear_x(data, shear, fill_value=128, interp_type=None):
    mt = fn.transforms.shear(shear=shear)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(mag_range=(0, 0.3), randomly_negate=True, mag_to_param=warp_y_param)
def shear_y(data, shear, fill_value=128, interp_type=None):
    mt = fn.transforms.shear(shear=shear)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(mag_range=(0.0, 1.0), randomly_negate=True, mag_to_param=warp_x_param)
def translate_x(data, rel_offset, shape, fill_value=128, interp_type=None):
    offset = rel_offset * shape[1]
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(
    mag_range=(0, 250), randomly_negate=True, mag_to_param=warp_x_param, name="translate_x"
)
def translate_x_no_shape(data, offset, fill_value=128, interp_type=None):
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(mag_range=(0.0, 1.0), randomly_negate=True, mag_to_param=warp_y_param)
def translate_y(data, rel_offset, shape, fill_value=128, interp_type=None):
    offset = rel_offset * shape[0]
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(
    mag_range=(0, 250), randomly_negate=True, mag_to_param=warp_y_param, name="translate_y"
)
def translate_y_no_shape(data, offset, fill_value=128, interp_type=None):
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(
        data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False
    )


@augmentation(mag_range=(0, 30), randomly_negate=True)
def rotate(data, angle, fill_value=128, interp_type=None, rotate_keep_size=True):
    return fn.rotate(
        data,
        angle=angle,
        fill_value=fill_value,
        interp_type=interp_type,
        keep_size=rotate_keep_size,
    )


def shift_enhance_range(magnitude):
    """The `enhance` operations (brightness, contrast, color, sharpness) accept magnitudes
    from [0, 2] range. However, the neutral magnitude is not 0 but 1 and the intuitive strength
    of the operation increases the further the magnitude is from 1. So, we specify magnitudes range
    to be in [0, 1] range, expect it to be randomly negated and then shift it by 1"""
    return 1 + magnitude


@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def brightness(data, parameter):
    return fn.brightness(data, brightness=parameter)


@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def contrast(data, parameter):
    """
    It follows PIL implementation of Contrast enhancement which uses a channel-weighted
    mean as a contrast center.
    """
    # assumes FHWC or HWC layout
    mean = fn.reductions.mean(data, axis_names="HW", keep_dims=True)
    rgb_weights = types.Constant(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    center = fn.reductions.sum(mean * rgb_weights, axis_names="C", keep_dims=True)
    # it could be just `fn.contrast(data, contrast=parameter, contrast_center=center)`
    # but for GPU `data` the `center` is in GPU mem, and that cannot be passed
    # as named arg (i.e. `contrast_center`) to the operator
    return fn.cast_like(center + (data - center) * parameter, data)


@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def color(data, parameter):
    return fn.saturation(data, saturation=parameter)


def sharpness_kernel(magnitude):
    # assumes magnitude: [-1, 1]
    blur = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13
    ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    return -magnitude * blur + (1 + magnitude) * ident


def sharpness_kernel_shifted(magnitude):
    # assumes magnitude: [0, 2]
    return sharpness_kernel(magnitude - 1)


@augmentation(
    mag_range=(0, 0.9), randomly_negate=True, mag_to_param=sharpness_kernel, param_device="auto"
)
def sharpness(data, kernel):
    """
    The outputs correspond to PIL's ImageEnhance.Sharpness with the exception for 1px
    border around the output. PIL computes convolution with smoothing filter only for
    valid positions (no out-of-bounds filter positions) and pads the output with the input.
    """
    return fn.experimental.filter(data, kernel)


def poster_mask_uint8(magnitude):
    # expects [0..8] where 8 yields identity mask and a 0
    # would be a mask that zeros all bits,
    # however, following the implementation for AA and RA referred
    # in the paper https://arxiv.org/pdf/1909.13719.pdf, we remap 0 to 1,
    # to avoid completely blank images
    magnitude = np.round(magnitude).astype(np.uint32)
    if magnitude <= 0:
        magnitude = 1
    elif magnitude > 8:
        magnitude = 8
    nbits = np.round(8 - magnitude).astype(np.uint32)
    removal_mask = np.uint8(2) ** nbits - 1
    return np.array(np.uint8(255) ^ removal_mask, dtype=np.uint8)


@augmentation(mag_range=(0, 4), mag_to_param=poster_mask_uint8, param_device="auto")
def posterize(data, mask):
    return data & mask


@augmentation(mag_range=(256, 0), param_device="auto")
def solarize(data, threshold):
    sample_inv = types.Constant(255, dtype=types.UINT8) - data
    mask_unchanged = data < threshold
    mask_inverted = mask_unchanged ^ True
    return mask_unchanged * data + mask_inverted * sample_inv


def solarize_add_shift(shift):
    if shift >= 128:
        raise Exception("The solarize_add augmentation accepts shifts from 0 to 128")
    return np.uint8(shift)


@augmentation(mag_range=(0, 110), param_device="auto", mag_to_param=solarize_add_shift)
def solarize_add(data, shift):
    mask_shifted = data < types.Constant(128, dtype=types.UINT8)
    mask_id = mask_shifted ^ True
    sample_shifted = data + shift
    return mask_shifted * sample_shifted + mask_id * data


@augmentation
def invert(data, _):
    return types.Constant(255, dtype=types.UINT8) - data


@augmentation
def equalize(data, _):
    """
    DALI's equalize follows OpenCV's histogram equalization.
    The PIL uses slightly different formula when transforming histogram's
    cumulative sum into lookup table.
    """
    return fn.experimental.equalize(data)


@augmentation
def auto_contrast(data, _):
    # assumes FHWC or HWC layout
    lo = fn.reductions.min(data, axis_names="HW", keep_dims=True)
    hi = fn.reductions.max(data, axis_names="HW", keep_dims=True)
    diff = hi - lo
    mask_scale = diff > 0
    mask_id = mask_scale ^ True
    # choose div so that scale ends up being 255 / diff if diff > 0 and 1 otherwise
    div_by = diff * mask_scale + types.Constant(255, dtype=types.UINT8) * mask_id
    scale = 255 / div_by
    scaled = (data - lo * mask_scale) * scale
    return fn.cast_like(scaled, data)


@augmentation
def identity(data, _):
    return data
