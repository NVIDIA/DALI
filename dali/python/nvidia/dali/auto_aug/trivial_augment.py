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

from typing import List, Optional, Tuple, Union

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core import _Augmentation, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs as _forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import (
    get_translations as _get_translations,
    pretty_select as _pretty_select,
)
from nvidia.dali.data_node import DataNode as _DataNode


def trivial_augment_wide(
    data: _DataNode,
    num_magnitude_bins: int = 31,
    shape: Optional[Union[_DataNode, Tuple[int, int]]] = None,
    fill_value: Optional[int] = 128,
    interp_type: Optional[types.DALIInterpType] = None,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
    seed: Optional[int] = None,
    excluded: Optional[List[str]] = None,
) -> _DataNode:
    """
    Applies TrivialAugment Wide (https://arxiv.org/abs/2103.10158) augmentation scheme to the
    provided batch of samples.

    Args
    ----
    data : DataNode
        A batch of samples to be processed. The supported samples are images
        of `HWC` layout and videos of `FHWC` layout, the supported data type is `uint8`.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    fill_value: int, optional
        A value to be used as a padding for images/frames transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images/frames are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    max_translate_abs: int or (int, int), optional
        Only valid when ``shapes`` is not provided. Specifies the maximal shift (in pixels)
        in the translation augmentation. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 32, which means the maximal magnitude
        shifts the image by 32 pixels.
    max_translate_rel: float or (float, float), optional
        Only valid when ``shapes`` argument is provided. Specifies the maximal shift as a
        fraction of image shape in the translation augmentations.
        If a tuple is specified, the first component limits the height, the second the width.
        Defaults to 1, which means the maximal magnitude shifts the image entirely out of
        the canvas.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    excluded: List[str], optional
        A list of names of the operations to be excluded from the default suite of augmentations.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tuned of the existing ones, you can use the
        :meth:`~nvidia.dali.auto_aug.trivial_augment.apply_trivial_augment` directly,
        which accepts a list of augmentations.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape
    augmentations = get_trivial_augment_wide_suite(
        use_shape=use_shape,
        max_translate_abs=max_translate_abs,
        max_translate_rel=max_translate_rel,
    )
    augmentation_names = set(aug.name for aug in augmentations)
    assert len(augmentation_names) == len(augmentations)
    excluded = excluded or []
    for name in excluded:
        if name not in augmentation_names:
            raise Exception(
                f"The `{name}` was specified in `excluded`, but the TrivialAugmentWide suite "
                f"does not contain augmentation with this name. "
                f"The augmentations in the suite are: {', '.join(augmentation_names)}."
            )
    selected_augments = [aug for aug in augmentations if aug.name not in excluded]
    return apply_trivial_augment(
        selected_augments, data, num_magnitude_bins=num_magnitude_bins, seed=seed, **aug_kwargs
    )


def apply_trivial_augment(
    augmentations: List[_Augmentation],
    data: _DataNode,
    num_magnitude_bins: int = 31,
    seed: Optional[int] = None,
    **kwargs,
) -> _DataNode:
    """
    Applies the list of `augmentations` in TrivialAugment
    (https://arxiv.org/abs/2103.10158) fashion.
    Each sample is processed with randomly selected transformation form `augmentations` list.
    The magnitude bin for every transformation is randomly selected from
    `[0, num_magnitude_bins - 1]`.

    Args
    ----
    augmentations : List[core._Augmentation]
        List of augmentations to be sampled and applied in TrivialAugment fashion.
    data : DataNode
        A batch of samples to be processed.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    kwargs:
        Any extra parameters to be passed when calling `augmentations`.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        TrivialAugment suite accept ``shapes``, ``fill_value`` and ``interp_type``.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if not isinstance(num_magnitude_bins, int) or num_magnitude_bins < 1:
        raise Exception(
            f"The `num_magnitude_bins` must be a positive integer, got {num_magnitude_bins}."
        )
    if len(augmentations) == 0:
        raise Exception(
            "The `augmentations` list cannot be empty. "
            "Got empty list in `apply_trivial_augment` call."
        )
    magnitude_bin = fn.random.uniform(
        values=list(range(num_magnitude_bins)), dtype=types.INT32, seed=seed
    )
    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    if use_signed_magnitudes:
        magnitude_bin = signed_bin(magnitude_bin, seed=seed)
    _forbid_unused_kwargs(augmentations, kwargs, "apply_trivial_augment")
    op_kwargs = dict(
        data=data, magnitude_bin=magnitude_bin, num_magnitude_bins=num_magnitude_bins, **kwargs
    )
    op_idx = fn.random.uniform(values=list(range(len(augmentations))), seed=seed, dtype=types.INT32)
    return _pretty_select(
        augmentations,
        op_idx,
        op_kwargs,
        auto_aug_name="apply_trivial_augment",
        ref_suite_name="get_trivial_augment_wide_suite",
    )


def get_trivial_augment_wide_suite(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> List[_Augmentation]:
    """
    Creates a list of 14 augmentations referred as wide augmentation space in TrivialAugment paper
    (https://arxiv.org/abs/2103.10158).

    Args
    ----
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image/frame shape.
        Useful if the samples processed with the auto augment have different shapes.
        If false, the offsets range is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If a tuple is specified, the first component limits height, the second the
        width. Defaults to 32.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image/frame
        shape in the translation augmentations. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 1.
    """
    default_translate_abs, default_translate_rel = 32, 1.0
    # translations = [translate_x, translate_y] with adjusted magnitude range
    translations = _get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )
    # [.augmentation((mag_low, mag_high), randomly_negate_mag, custom_magnitude_to_param_mapping]
    return translations + [
        a.shear_x.augmentation((0, 0.99), True),
        a.shear_y.augmentation((0, 0.99), True),
        a.rotate.augmentation((0, 135), True),
        a.brightness.augmentation((0.01, 0.99), True, a.shift_enhance_range),
        a.contrast.augmentation((0.01, 0.99), True, a.shift_enhance_range),
        a.color.augmentation((0.01, 0.99), True, a.shift_enhance_range),
        a.sharpness.augmentation((0.01, 0.99), True, a.sharpness_kernel),
        a.posterize.augmentation((8, 2), False, a.poster_mask_uint8),
        # solarization strength increases with decreasing magnitude (threshold)
        a.solarize.augmentation((256, 0)),
        a.equalize,
        a.auto_contrast,
        a.identity,
    ]
