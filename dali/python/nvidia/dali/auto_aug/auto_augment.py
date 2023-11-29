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
from nvidia.dali.auto_aug.core import _Augmentation, Policy, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs as _forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import (
    get_translations as _get_translations,
    pretty_select as _pretty_select,
)
from nvidia.dali.data_node import DataNode as _DataNode

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples."
    )


def auto_augment(
    data: _DataNode,
    policy_name: str = "image_net",
    shape: Optional[Union[_DataNode, Tuple[int, int]]] = None,
    fill_value: Optional[int] = 128,
    interp_type: Optional[types.DALIInterpType] = None,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
    seed: Optional[int] = None,
) -> _DataNode:
    """
    Applies one of the predefined policies from the AutoAugment
    paper (https://arxiv.org/abs/1805.09501) to the provided batch of samples.

    Args
    ----
    data : DataNode
        A batch of samples to be processed. The supported samples are images
        of `HWC` layout and videos of `FHWC` layout, the supported data type is `uint8`.
    policy_name : str, optional
        The name of predefined policy. Acceptable values are: `image_net`,
        `reduced_image_net`, `svhn`, `reduced_cifar10`. Defaults to `image_net`.
    shape: DataNode or Tuple[int, int], optional
        The size (height and width) of the image or frames in the video sequence
        passed as the `data`. If specified, the magnitude of `translation` operations
        depends on the image/frame shape and spans from 0 to `max_translate_rel * shape`.
        Otherwise, the magnitude range is `[0, max_translate_abs]` for any sample.
    fill_value: int, optional
        A value to be used as a padding for images/frames transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images/frames are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    max_translate_abs: int or (int, int), optional
        Only valid when `shape` is not provided. Specifies the maximal shift (in pixels)
        in the translation augmentation. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 250, which means the maximal magnitude
        shifts the image by 250 pixels.
    max_translate_rel: float or (float, float), optional
        Only valid when `shape` argument is provided. Specifies the maximal shift as a
        fraction of image shape in the translation augmentations.
        If a tuple is specified, the first component limits the height, the second the width.
        Defaults to 1, which means the maximal magnitude shifts the image entirely out of
        the canvas.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    predefined_policies = {
        "image_net": get_image_net_policy,
        "reduced_image_net": get_reduced_image_net_policy,
        "svhn": get_svhn_policy,
        "reduced_cifar10": get_reduced_cifar10_policy,
    }
    policies_without_translation = ("reduced_image_net",)
    shape_related_args = (
        (shape, "shape"),
        (max_translate_abs, "max_translate_abs"),
        (max_translate_rel, "max_translate_rel"),
    )
    if not isinstance(policy_name, str) or policy_name not in predefined_policies:
        policies_str = ", ".join([f"`{name}`" for name in predefined_policies.keys()])
        raise Exception(
            f"The `policy_name` must be a string that takes one of the values: {policies_str}"
        )
    if policy_name in policies_without_translation:
        shape_arg = next((name for arg, name in shape_related_args if arg is not None), None)
        if shape_arg is not None:
            raise Exception(
                f"The policy `{policy_name}` does not contain any augmentations that rely on the "
                f"image shape. The `{shape_arg}` argument must not be specified in that case."
            )

    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape

    if policy_name in policies_without_translation:
        policy = predefined_policies[policy_name]()
    else:
        policy = predefined_policies[policy_name](
            use_shape=use_shape,
            max_translate_abs=max_translate_abs,
            max_translate_rel=max_translate_rel,
        )

    return apply_auto_augment(policy, data, seed, **aug_kwargs)


def auto_augment_image_net(
    data: _DataNode,
    shape: Optional[Union[_DataNode, Tuple[int, int]]] = None,
    fill_value: Optional[int] = 128,
    interp_type: Optional[types.DALIInterpType] = None,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
    seed: Optional[int] = None,
) -> _DataNode:
    """
    Applies `image_net_policy` in AutoAugment (https://arxiv.org/abs/1805.09501)
    fashion to the provided batch of samples.

    Equivalent to :meth:`~nvidia.dali.auto_aug.auto_augment.auto_augment` call with ``policy_name``
    specified to ``'image_net'``.
    See :meth:`~nvidia.dali.auto_aug.auto_augment.auto_augment` function for details.
    """
    return auto_augment(
        data,
        "image_net",
        shape,
        fill_value,
        interp_type,
        max_translate_abs,
        max_translate_rel,
        seed,
    )


def apply_auto_augment(
    policy: Policy, data: _DataNode, seed: Optional[int] = None, **kwargs
) -> _DataNode:
    """
    Applies AutoAugment (https://arxiv.org/abs/1805.09501) augmentation scheme to the
    provided batch of samples.

    Args
    ----
    policy: Policy
        Set of sequences of augmentations to be applied in AutoAugment fashion.
    data : DataNode
        A batch of samples to be processed.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    kwargs:
        A dictionary of extra parameters to be passed when calling augmentations.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        AutoAugment suite accept ``shape``, ``fill_value`` and ``interp_type``.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if len(policy.sub_policies) == 0:
        raise Exception(f"Cannot run empty policy. Got {policy} in `apply_auto_augment` call.")
    max_policy_len = max(len(sub_policy) for sub_policy in policy.sub_policies)
    should_run = fn.random.uniform(
        range=[0, 1], shape=(max_policy_len,), dtype=types.FLOAT, seed=seed
    )
    sub_policy_id = fn.random.uniform(
        values=list(range(len(policy.sub_policies))), seed=seed, dtype=types.INT32
    )
    run_probabilities = _sub_policy_to_probability_map(policy)[sub_policy_id]
    magnitude_bins = _sub_policy_to_magnitude_bin_map(policy)[sub_policy_id]
    aug_ids, augmentations = _sub_policy_to_augmentation_map(policy)
    aug_ids = aug_ids[sub_policy_id]
    if any(aug.randomly_negate for aug in policy.augmentations.values()):
        magnitude_bins = signed_bin(magnitude_bins, seed=seed, shape=(max_policy_len,))
    _forbid_unused_kwargs(policy.augmentations.values(), kwargs, "apply_auto_augment")
    for stage_id in range(max_policy_len):
        if should_run[stage_id] < run_probabilities[stage_id]:
            op_kwargs = dict(
                data=data,
                magnitude_bin=magnitude_bins[stage_id],
                num_magnitude_bins=policy.num_magnitude_bins,
                **kwargs,
            )
            data = _pretty_select(
                augmentations[stage_id],
                aug_ids[stage_id],
                op_kwargs,
                auto_aug_name="apply_auto_augment",
                ref_suite_name="get_image_net_policy",
            )
    return data


def get_image_net_policy(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> Policy:
    """
    Creates augmentation policy tuned for the ImageNet as described in
    AutoAugment paper (https://arxiv.org/abs/1805.09501).
    The returned policy can be run with
    :meth:`~nvidia.dali.auto_aug.auto_augment.apply_auto_augment`.

    Args
    ----
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image/frame shape.
        Useful if the samples processed with the auto augment have different shapes.
        If false, the offsets range is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If a tuple is specified, the first component limits height, the second the
        width. Defaults to 250.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image/frame
        shape in the translation augmentations. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 1.
    """
    default_translate_abs, default_translate_rel = 250, 1.0
    _, translate_y = _get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )
    shear_x = a.shear_x.augmentation((0, 0.3), True)
    shear_y = a.shear_y.augmentation((0, 0.3), True)
    rotate = a.rotate.augmentation((0, 30), True)
    color = a.color.augmentation((0.1, 1.9), False, None)
    posterize = a.posterize.augmentation((0, 4), False, a.poster_mask_uint8)
    solarize = a.solarize.augmentation((0, 256), False)
    solarize_add = a.solarize_add.augmentation((0, 110), False)
    invert = a.invert
    equalize = a.equalize
    auto_contrast = a.auto_contrast
    return Policy(
        name="ImageNetPolicy",
        num_magnitude_bins=11,
        sub_policies=[
            [(equalize, 0.8, None), (shear_y, 0.8, 4)],
            [(color, 0.4, 9), (equalize, 0.6, None)],
            [(color, 0.4, 1), (rotate, 0.6, 8)],
            [(solarize, 0.8, 3), (equalize, 0.4, None)],
            [(solarize, 0.4, 2), (solarize, 0.6, 2)],
            [(color, 0.2, 0), (equalize, 0.8, None)],
            [(equalize, 0.4, None), (solarize_add, 0.8, 3)],
            [(shear_x, 0.2, 9), (rotate, 0.6, 8)],
            [(color, 0.6, 1), (equalize, 1.0, None)],
            [(invert, 0.4, None), (rotate, 0.6, 0)],
            [(equalize, 1.0, None), (shear_y, 0.6, 3)],
            [(color, 0.4, 7), (equalize, 0.6, None)],
            [(posterize, 0.4, 6), (auto_contrast, 0.4, None)],
            [(solarize, 0.6, 8), (color, 0.6, 9)],
            [(solarize, 0.2, 4), (rotate, 0.8, 9)],
            [(rotate, 1.0, 7), (translate_y, 0.8, 9)],
            [(solarize, 0.8, 4)],
            [(shear_y, 0.8, 0), (color, 0.6, 4)],
            [(color, 1.0, 0), (rotate, 0.6, 2)],
            [(equalize, 0.8, None)],
            [(equalize, 1.0, None), (auto_contrast, 0.6, None)],
            [(shear_y, 0.4, 7), (solarize_add, 0.6, 7)],
            [(posterize, 0.8, 2), (solarize, 0.6, 10)],
            [(solarize, 0.6, 8), (equalize, 0.6, None)],
            [(color, 0.8, 6), (rotate, 0.4, 5)],
        ],
    )


def get_reduced_cifar10_policy(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> Policy:
    """
    Creates augmentation policy tuned with the reduced CIFAR-10 as described
    in AutoAugment paper (https://arxiv.org/abs/1805.09501).
    The returned policy can be run with
    :meth:`~nvidia.dali.auto_aug.auto_augment.apply_auto_augment`.

    Args
    ----
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image/frame shape.
        Useful if the samples processed with the auto augment have different shapes.
        If false, the offsets range is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If a tuple is specified, the first component limits height, the second the
        width. Defaults to 250.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image/frame
        shape in the translation augmentations. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 1.
    """
    default_translate_abs, default_translate_rel = 250, 1.0
    translate_x, translate_y = _get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )
    shear_y = a.shear_y.augmentation((0, 0.3), True)
    rotate = a.rotate.augmentation((0, 30), True)
    brightness = a.brightness.augmentation((0.1, 1.9), False, None)
    color = a.color.augmentation((0.1, 1.9), False, None)
    contrast = a.contrast.augmentation((0.1, 1.9), False, None)
    sharpness = a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted)
    posterize = a.posterize.augmentation((0, 4), False, a.poster_mask_uint8)
    solarize = a.solarize.augmentation((0, 256), False)
    invert = a.invert
    equalize = a.equalize
    auto_contrast = a.auto_contrast
    return Policy(
        name="ReducedCifar10Policy",
        num_magnitude_bins=11,
        sub_policies=[
            [(invert, 0.1, None), (contrast, 0.2, 6)],
            [(rotate, 0.7, 2), (translate_x, 0.3, 9)],
            [(sharpness, 0.8, 1), (sharpness, 0.9, 3)],
            [(shear_y, 0.5, 8), (translate_y, 0.7, 9)],
            [(auto_contrast, 0.5, None), (equalize, 0.9, None)],
            [(shear_y, 0.2, 7), (posterize, 0.3, 7)],
            [(color, 0.4, 3), (brightness, 0.6, 7)],
            [(sharpness, 0.3, 9), (brightness, 0.7, 9)],
            [(equalize, 0.6, None), (equalize, 0.5, None)],
            [(contrast, 0.6, 7), (sharpness, 0.6, 5)],
            [(color, 0.7, 7), (translate_x, 0.5, 8)],
            [(equalize, 0.3, None), (auto_contrast, 0.4, None)],
            [(translate_y, 0.4, 3), (sharpness, 0.2, 6)],
            [(brightness, 0.9, 6), (color, 0.2, 8)],
            [(solarize, 0.5, 2)],
            [(equalize, 0.2, None), (auto_contrast, 0.6, None)],
            [(equalize, 0.2, None), (equalize, 0.6, None)],
            [(color, 0.9, 9), (equalize, 0.6, None)],
            [(auto_contrast, 0.8, None), (solarize, 0.2, 8)],
            [(brightness, 0.1, 3), (color, 0.7, 0)],
            [(solarize, 0.4, 5), (auto_contrast, 0.9, None)],
            [(translate_y, 0.9, 9), (translate_y, 0.7, 9)],
            [(auto_contrast, 0.9, None), (solarize, 0.8, 3)],
            [(equalize, 0.8, None), (invert, 0.1, None)],
            [(translate_y, 0.7, 9), (auto_contrast, 0.9, None)],
        ],
    )


def get_svhn_policy(
    use_shape: bool = False,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
) -> Policy:
    """
    Creates augmentation policy tuned with the SVHN as described
    in AutoAugment paper (https://arxiv.org/abs/1805.09501).
    The returned policy can be run with
    :meth:`~nvidia.dali.auto_aug.auto_augment.apply_auto_augment`.

    Args
    ----
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image/frame shape.
        Useful if the samples processed with the auto augment have different shapes.
        If false, the offsets range is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If a tuple is specified, the first component limits height, the second the
        width. Defaults to 250.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image/frame
        shape in the translation augmentations. If a tuple is specified, the first component limits
        height, the second the width. Defaults to 1.
    """
    default_translate_abs, default_translate_rel = 250, 1.0
    translate_x, translate_y = _get_translations(
        use_shape,
        default_translate_abs,
        default_translate_rel,
        max_translate_abs,
        max_translate_rel,
    )
    shear_x = a.shear_x.augmentation((0, 0.3), True)
    shear_y = a.shear_y.augmentation((0, 0.3), True)
    rotate = a.rotate.augmentation((0, 30), True)
    contrast = a.contrast.augmentation((0.1, 1.9), False, None)
    solarize = a.solarize.augmentation((0, 256), False)
    invert = a.invert
    equalize = a.equalize
    auto_contrast = a.auto_contrast
    return Policy(
        name="SvhnPolicy",
        num_magnitude_bins=11,
        sub_policies=[
            [(shear_x, 0.9, 4), (invert, 0.2, None)],
            [(shear_y, 0.9, 8), (invert, 0.7, None)],
            [(equalize, 0.6, None), (solarize, 0.6, 6)],
            [(invert, 0.9, None), (equalize, 0.6, None)],
            [(equalize, 0.6, None), (rotate, 0.9, 3)],
            [(shear_x, 0.9, 4), (auto_contrast, 0.8, None)],
            [(shear_y, 0.9, 8), (invert, 0.4, None)],
            [(shear_y, 0.9, 5), (solarize, 0.2, 6)],
            [(invert, 0.9, None), (auto_contrast, 0.8, None)],
            [(equalize, 0.6, None), (rotate, 0.9, 3)],
            [(shear_x, 0.9, 4), (solarize, 0.3, 3)],
            [(shear_y, 0.8, 8), (invert, 0.7, None)],
            [(equalize, 0.9, None), (translate_y, 0.6, 6)],
            [(invert, 0.9, None), (equalize, 0.6, None)],
            [(contrast, 0.3, 3), (rotate, 0.8, 4)],
            [(invert, 0.8, None)],
            [(shear_y, 0.7, 6), (solarize, 0.4, 8)],
            [(invert, 0.6, None), (rotate, 0.8, 4)],
            [(shear_y, 0.3, 7), (translate_x, 0.9, 3)],
            [(shear_x, 0.1, 6), (invert, 0.6, None)],
            [(solarize, 0.7, 2), (translate_y, 0.6, 7)],
            [(shear_y, 0.8, 4), (invert, 0.8, None)],
            [(shear_x, 0.7, 9), (translate_y, 0.8, 3)],
            [(shear_y, 0.8, 5), (auto_contrast, 0.7, None)],
            [(shear_x, 0.7, 2), (invert, 0.1, None)],
        ],
    )


def get_reduced_image_net_policy() -> Policy:
    """
    Creates augmentation policy tuned with the reduced ImageNet as described in
    AutoAugment paper (https://arxiv.org/abs/1805.09501).
    The returned policy can be run with
    :meth:`~nvidia.dali.auto_aug.auto_augment.apply_auto_augment`.
    """
    shear_x = a.shear_x.augmentation((0, 0.3), True)
    rotate = a.rotate.augmentation((0, 30), True)
    color = a.color.augmentation((0.1, 1.9), False, None)
    contrast = a.contrast.augmentation((0.1, 1.9), False, None)
    sharpness = a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted)
    posterize = a.posterize.augmentation((0, 4), False, a.poster_mask_uint8)
    solarize = a.solarize.augmentation((0, 256), False)
    invert = a.invert
    equalize = a.equalize
    auto_contrast = a.auto_contrast
    return Policy(
        name="ReducedImageNetPolicy",
        num_magnitude_bins=11,
        sub_policies=[
            [(posterize, 0.4, 8), (rotate, 0.6, 9)],
            [(solarize, 0.6, 5), (auto_contrast, 0.6, None)],
            [(equalize, 0.8, None), (equalize, 0.6, None)],
            [(posterize, 0.6, 7), (posterize, 0.6, 6)],
            [(equalize, 0.4, None), (solarize, 0.2, 4)],
            [(equalize, 0.4, None), (rotate, 0.8, 8)],
            [(solarize, 0.6, 3), (equalize, 0.6, None)],
            [(posterize, 0.8, 5), (equalize, 1.0, None)],
            [(rotate, 0.2, 3), (solarize, 0.6, 8)],
            [(equalize, 0.6, None), (posterize, 0.4, 6)],
            [(rotate, 0.8, 8), (color, 0.4, 0)],
            [(rotate, 0.4, 9), (equalize, 0.6, None)],
            [(equalize, 0.8, None)],
            [(invert, 0.6, None), (equalize, 1.0, None)],
            [(color, 0.6, 4), (contrast, 1.0, 8)],
            [(rotate, 0.8, 8), (color, 1.0, 2)],
            [(color, 0.8, 8), (solarize, 0.8, 7)],
            [(sharpness, 0.4, 7), (invert, 0.6, None)],
            [(shear_x, 0.6, 5), (equalize, 1.0, None)],
            [(color, 0.4, 0), (equalize, 0.6, None)],
            [(equalize, 0.4, None), (solarize, 0.2, 4)],
            [(solarize, 0.6, 5), (auto_contrast, 0.6, None)],
            [(invert, 0.6, None), (equalize, 1.0, None)],
            [(color, 0.6, 4), (contrast, 1.0, 8)],
            [(equalize, 0.8, None), (equalize, 0.6, None)],
        ],
    )


def _sub_policy_to_probability_map(policy: Policy) -> _DataNode:
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    prob = np.array(
        [[0.0 for _ in range(max_policy_len)] for _ in range(len(sub_policies))], dtype=np.float32
    )
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            prob[sub_policy_id, stage_idx] = p
    return types.Constant(prob)


def _sub_policy_to_magnitude_bin_map(policy: Policy) -> _DataNode:
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    magnitude_bin = np.array(
        [[0 for _ in range(max_policy_len)] for _ in range(len(sub_policies))], dtype=np.int32
    )
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            # use dummy value instead of None, it will be ignored anyway
            val = mag if mag is not None else -999
            magnitude_bin[sub_policy_id, stage_idx] = val
    return types.Constant(magnitude_bin)


def _sub_policy_to_augmentation_matrix_map(
    policy: Policy,
) -> Tuple[np.ndarray, List[List[_Augmentation]]]:
    """
    Creates a matrix of operators to be called for given sub policy at given stage.
    The output is a tuple `(m, augments)`, where `augments` is a list of augmentations per stage
    - each entry contains a reduced list of unique augmentations used in a corresponding stage.
    The `m` matrix contains the mapping from the original sub_policy_id, to the index within the
    reduced list, for every stage. I.e., for policy `sub_policy_idx`, as the `stage_idx`-ith
    operation in a sequence, the `augments[stage_idx][m[sub_policy_idx][stage_idx]]` operator
    should be called.
    """
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    augmentations = []  # list of augmentations in each stage
    for stage_idx in range(max_policy_len):
        stage_augments = set()
        stage_augments_list = []
        for sub_policy in sub_policies:
            if stage_idx < len(sub_policy):
                aug, _, _ = sub_policy[stage_idx]
                if aug not in stage_augments:
                    stage_augments.add(aug)
                    stage_augments_list.append(aug)
        augmentations.append(stage_augments_list + [a.identity])
    identity_id = [len(stage_augments) - 1 for stage_augments in augmentations]
    augment_to_id = [
        {augmentation: i for i, augmentation in enumerate(stage_augments)}
        for stage_augments in augmentations
    ]
    augments_by_id = np.array(
        [
            [identity_id[stage_idx] for stage_idx in range(max_policy_len)]
            for _ in range(len(sub_policies))
        ],
        dtype=np.int32,
    )
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (augment, p, mag) in enumerate(sub_policy):
            augments_by_id[sub_policy_id, stage_idx] = augment_to_id[stage_idx][augment]
    return augments_by_id, augmentations


def _sub_policy_to_augmentation_map(policy: Policy) -> Tuple[_DataNode, List[List[_Augmentation]]]:
    matrix, augments = _sub_policy_to_augmentation_matrix_map(policy)
    return types.Constant(matrix), augments
