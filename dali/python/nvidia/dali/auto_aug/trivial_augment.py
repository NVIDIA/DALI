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

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core.utils import select

trivial_augment_wide_suite = {
    "shear_x": a.shear_x.augmentation((0, 0.99), True),
    "shear_y": a.shear_y.augmentation((0, 0.99), True),
    "translate_x": a.translate_x_no_shape.augmentation((0, 32), True),
    "translate_y": a.translate_y_no_shape.augmentation((0, 32), True),
    "rotate": a.rotate.augmentation((0, 135), True),
    "brightness": a.brightness.augmentation((0.01, 0.99), True, a.shift_enhance_range),
    "contrast": a.contrast.augmentation((0.01, 0.99), True, a.shift_enhance_range),
    "color": a.color.augmentation((0.01, 0.99), True, a.shift_enhance_range),
    "sharpness": a.sharpness.augmentation((0.01, 0.99), True, a.sharpness_kernel),
    "posterize": a.posterize.augmentation((8, 2), False, a.poster_mask_uint8),
    "solarize": a.solarize.augmentation((256, 0)),
    "equalize": a.equalize,
    "auto_contrast": a.auto_contrast,
    "identity": a.identity,
}


def trivial_augment_wide(samples, num_magnitude_bins=31, fill_value=0, interp_type=None, seed=None,
                         excluded=None):
    """
    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the
    provided batch of samples.

    Parameter
    ---------
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    fill_value: int, optional
        A value to be used as a padding for images transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    excluded: List[str], optional
        A list of names of the operations to be excluded from the default suite of augmentations.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tuned of the existing ones, you can use the `apply_trivial_augment`
        directly, which accepts a list of augmentations.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    augments = dict(**trivial_augment_wide_suite)
    kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    excluded = excluded or tuple()
    for name in excluded:
        if name not in augments:
            raise Exception(
                f"The `{name}` was specified in `excluded`, but the trivial_augment_wide_suite "
                f"does not contain such an augmentation.")
    selected_augments = [augment for name, augment in augments.items() if name not in excluded]
    return apply_trivial_augment(selected_augments, samples, num_magnitude_bins=num_magnitude_bins,
                                 seed=seed, **kwargs)


def apply_trivial_augment(augmentations, samples, num_magnitude_bins=31, seed=None, **kwargs):
    """
    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the
    provided batch of samples but with a custom set of augmentations.

    Parameter
    ---------
    augmentations : List[core.Augmentation]
        List of augmentations to be sampled and applied in TrivialAugment fashion.
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    excluded: List[str], optional
        A list of names of the operations to be excluded from the `rand_augment_suite`.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tuned of the existing ones, you can use the `apply_rand_augment`
        directly, which accepts a list of augmentations.
    kwargs:
        Any extra parameters to be passed when calling `augmentations`.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        random augment suite accept `shapes`, `fill_value` and `interp_type`.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if num_magnitude_bins <= 1:
        raise Exception(
            f"The number of magnitude bins cannot be less than 1, got {num_magnitude_bins}.")
    if len(augmentations) == 0:
        return samples
    magnitude_bin_idx = fn.random.uniform(range=[0, num_magnitude_bins - 1], dtype=types.INT32,
                                          seed=seed)
    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    if not use_signed_magnitudes:
        random_sign = None
    else:
        random_sign = fn.random.uniform(range=[0, 1], dtype=types.INT32, seed=seed)
    op_kwargs = dict(samples=samples, magnitude_bin_idx=magnitude_bin_idx,
                     num_magnitude_bins=num_magnitude_bins, random_sign=random_sign,
                     **kwargs)
    op_idx = fn.random.uniform(values=list(range(len(augmentations))), seed=seed, dtype=types.INT32)
    return select(augmentations, op_idx, op_kwargs)
