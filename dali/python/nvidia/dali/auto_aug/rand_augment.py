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

import warnings

from typing import List, Optional

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core import signed_bin, _Augmentation
from nvidia.dali.auto_aug.core._args import \
    forbid_unused_kwargs as _forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import \
    get_translations as _get_translations, \
    pretty_select as _pretty_select
from nvidia.dali.data_node import DataNode as _DataNode


def rand_augment(data: _DataNode, n: int, m: int, num_magnitude_bins: int = 31,
                 shape: Optional[_DataNode] = None, fill_value: Optional[int] = 128,
                 interp_type: Optional[types.DALIInterpType] = None,
                 max_translate_abs: Optional[int] = None, max_translate_rel: Optional[float] = None,
                 seed: Optional[int] = None, monotonic_mag: bool = True,
                 excluded: Optional[List[str]] = None) -> _DataNode:
    """
    Applies RandAugment (https://arxiv.org/abs/1909.13719) augmentation scheme to the
    provided batch of samples.

    Args
    ----
    data : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude (strength) of each operation to be applied, it must be an integer
        within ``[0, num_magnitude_bins - 1]``.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    shape: DataNode, optional
        A batch of shapes of the samples. If specified, the magnitude of `translation`
        operations depends on the image shape and spans from 0 to ``max_translate_rel * shape``.
        Otherwise, the magnitude range is ``[0, max_translate_abs]`` for any sample.
    fill_value: int, optional
        A value to be used as a padding for images transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    max_translate_abs: int or (int, int), optional
        Only valid when ``shapes`` is not provided. Specifies the maximal shift (in pixels)
        in the translation augmentation. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 100, which means the maximal magnitude
        shifts the image by 100 pixels.
    max_translate_rel: float or (float, float), optional
        Only valid when ``shapes`` argument is provided. Specifies the maximal shift as a
        fraction of image shape in the translation augmentations.
        If a tuple is specified, the first component limits the height, the second the width.
        Defaults to around `0.45` (100/224).
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    monotonic_mag: bool, optional
        There are two flavours of RandAugment available in different frameworks. For the default
        ``monotonic_mag=True`` the strength of operations that accept magnitude bins increases with
        the increasing bins. If set to False, the magnitude ranges for some color operations differ.
        There, the :meth:`~nvidia.dali.auto_aug.augmentations.posterize` and
        :meth:`~nvidia.dali.auto_aug.augmentations.solarize` strength decreases with increasing
        magnitude bins and enhance operations (
        :meth:`~nvidia.dali.auto_aug.augmentations.brightness`,
        :meth:`~nvidia.dali.auto_aug.augmentations.contrast`,
        :meth:`~nvidia.dali.auto_aug.augmentations.color`,
        :meth:`~nvidia.dali.auto_aug.augmentations.sharpness`) use (0.1, 1.9) range,
        which means that the strength decreases the closer the magnitudes are to the center
        of the range. See
        :meth:`~nvidia.dali.auto_aug.rand_augment.get_rand_augment_non_monotonic_suite`.
    excluded: List[str], optional
        A list of names of the operations to be excluded from the default suite of augmentations.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tune the existing ones, you can use the
        :meth:`~nvidia.dali.auto_aug.rand_augment.apply_rand_augment` directly, which accepts
        a list of augmentations.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape
    if monotonic_mag:
        augmentations = get_rand_augment_suite(use_shape, max_translate_abs, max_translate_rel)
    else:
        augmentations = get_rand_augment_non_monotonic_suite(use_shape, max_translate_abs,
                                                             max_translate_rel)
    augmentation_names = set(aug.name for aug in augmentations)
    assert len(augmentation_names) == len(augmentations)
    excluded = excluded or []
    for name in excluded:
        if name not in augmentation_names:
            raise Exception(f"The `{name}` was specified in `excluded`, but the RandAugment suite "
                            f"does not contain augmentation with this name. "
                            f"The augmentations in the suite are: {', '.join(augmentation_names)}.")
    selected_augments = [aug for aug in augmentations if aug.name not in excluded]
    return apply_rand_augment(selected_augments, data, n, m,
                              num_magnitude_bins=num_magnitude_bins, seed=seed, **aug_kwargs)


def apply_rand_augment(augmentations: List[_Augmentation], data: _DataNode, n: int, m: int,
                       num_magnitude_bins: int = 31, seed: Optional[int] = None,
                       **kwargs) -> _DataNode:
    """
    Applies the list of ``augmentations`` in RandAugment (https://arxiv.org/abs/1909.13719) fashion.
    Each sample is transformed with ``n`` operations in a sequence randomly selected from the
    ``augmentations`` list. Each operation uses ``m`` as the magnitude bin.

    Args
    ----
    augmentations : List[core._Augmentation]
        List of augmentations to be sampled and applied in RandAugment fashion.
    data : DataNode
        A batch of samples to be processed.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude bin (strength) of each operation to be applied, it must be an integer
        within ``[0, num_magnitude_bins - 1]``.
    num_magnitude_bins: int
        The number of bins to divide the magnitude ranges into.
    seed: int
        Seed to be used to randomly sample operations (and to negate magnitudes).
    kwargs:
        Any extra parameters to be passed when calling `augmentations`.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        RandAugment suite accept ``shapes``, ``fill_value`` and ``interp_type``.
    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if not isinstance(n, int) or n < 0:
        raise Exception(
            f"The number of operations to apply `n` must be a non-negative integer, got {n}.")
    if not isinstance(num_magnitude_bins, int) or num_magnitude_bins < 1:
        raise Exception(
            f"The `num_magnitude_bins` must be a positive integer, got {num_magnitude_bins}.")
    if not isinstance(m, int) or not 0 <= m < num_magnitude_bins:
        raise Exception(f"The magnitude bin `m` must be an integer from "
                        f"`[0, {num_magnitude_bins - 1}]` range. Got {m}.")
    if n == 0:
        warnings.warn(
            "The `apply_rand_augment` was called with `n=0`, "
            "no augmentation will be applied.", Warning)
        return data
    if len(augmentations) == 0:
        raise Exception("The `augmentations` list cannot be empty, unless n=0. "
                        "Got empty list in `apply_rand_augment` call.")
    shape = tuple() if n == 1 else (n, )
    op_idx = fn.random.uniform(values=list(range(len(augmentations))), seed=seed, shape=shape,
                               dtype=types.INT32)
    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    mag_bin = signed_bin(m, seed=seed, shape=shape) if use_signed_magnitudes else m
    _forbid_unused_kwargs(augmentations, kwargs, 'apply_rand_augment')
    for level_idx in range(n):
        level_mag_bin = mag_bin if not use_signed_magnitudes or n == 1 else mag_bin[level_idx]
        op_kwargs = dict(data=data, magnitude_bin=level_mag_bin,
                         num_magnitude_bins=num_magnitude_bins, **kwargs)
        level_op_idx = op_idx if n == 1 else op_idx[level_idx]
        data = _pretty_select(augmentations, level_op_idx, op_kwargs,
                              auto_aug_name='apply_rand_augment',
                              ref_suite_name='get_rand_augment_suite')
    return data


def get_rand_augment_suite(use_shape: bool = False, max_translate_abs: Optional[int] = None,
                           max_translate_rel: Optional[float] = None) -> List[_Augmentation]:
    """
    Creates a list of RandAugment augmentations.

    Args
    ----
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image. Useful if the
        images processed with the auto augment have different shapes. If false, the offsets range
        is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If a tuple is specified, the first component limits height, the second the
        width. Defaults 100.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image shape
        in the translation augmentations. If a tuple is specified, the first component limits
        height, the second the width. Defaults to around `0.45` (100/224).
    """
    default_translate_abs, default_translate_rel = 100, 100 / 224
    # translations = [translate_x, translate_y] with adjusted magnitude range
    translations = _get_translations(use_shape, default_translate_abs, default_translate_rel,
                                     max_translate_abs, max_translate_rel)
    # [.augmentation((mag_low, mag_high), randomly_negate_mag, magnitude_to_param_custom_mapping]
    return translations + [
        a.shear_x.augmentation((0, 0.3), True),
        a.shear_y.augmentation((0, 0.3), True),
        a.rotate.augmentation((0, 30), True),
        a.brightness.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.contrast.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.color.augmentation((0, 0.9), True, a.shift_enhance_range),
        a.sharpness.augmentation((0, 0.9), True, a.sharpness_kernel),
        a.posterize.augmentation((8, 4), False, a.poster_mask_uint8),
        # solarization strength increases with decreasing magnitude (threshold)
        a.solarize.augmentation((256, 0)),
        a.equalize,
        a.auto_contrast,
        a.identity,
    ]


def get_rand_augment_non_monotonic_suite(
        use_shape: bool = False, max_translate_abs: Optional[int] = None,
        max_translate_rel: Optional[float] = None) -> List[_Augmentation]:
    """
    Similarly to :meth:`~nvidia.dali.auto_aug.rand_augment.get_rand_augment_suite` creates a list
    of RandAugment augmentations.

    This variant uses brightness, contrast, color, sharpness, posterize, and solarize
    with magnitude ranges as used by the AutoAugment. However, those ranges do not meet
    the intuition that the bigger magnitude bin corresponds to stronger operation.
    """
    default_translate_abs, default_translate_rel = 100, 100 / 224
    # translations = [translate_x, translate_y] with adjusted magnitude range
    translations = _get_translations(use_shape, default_translate_abs, default_translate_rel,
                                     max_translate_abs, max_translate_rel)
    return translations + [
        a.shear_x.augmentation((0, 0.3), True),
        a.shear_y.augmentation((0, 0.3), True),
        a.rotate.augmentation((0, 30), True),
        a.brightness.augmentation((0.1, 1.9), False, None),
        a.contrast.augmentation((0.1, 1.9), False, None),
        a.color.augmentation((0.1, 1.9), False, None),
        a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
        a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
        a.solarize.augmentation((0, 256), False, None),
        a.equalize,
        a.auto_contrast,
        a.identity,
    ]
