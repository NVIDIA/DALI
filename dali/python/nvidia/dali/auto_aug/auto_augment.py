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

from typing import List, Optional, Tuple

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core import _Augmentation, Policy, signed_bin
from nvidia.dali.auto_aug.core._args import forbid_unused_kwargs as _forbid_unused_kwargs
from nvidia.dali.auto_aug.core._utils import \
    parse_validate_offset as _parse_validate_offset, \
    pretty_select as _pretty_select
from nvidia.dali.data_node import DataNode as _DataNode

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


def auto_augment_image_net(sample: _DataNode, shape: Optional[_DataNode] = None,
                           fill_value: Optional[int] = 128,
                           interp_type: Optional[types.DALIInterpType] = None,
                           max_translate_abs: Optional[int] = None,
                           max_translate_rel: Optional[float] = None, seed: Optional[int] = None):
    """
    Applies `auto_augment_image_net_policy` in AutoAugment (https://arxiv.org/abs/1805.09501)
    fashion to the provided batch of samples.

    Parameter
    ---------
    sample : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    shapes: DataNode, optional
        A batch of shapes of the `sample`. If specified, the magnitude of `translation`
        operations depends on the image shape and spans from 0 to `max_translate_rel * shape`.
        Otherwise, the magnitude range is `[0, max_translate_abs]` for any sample.
    fill_value: int, optional
        A value to be used as a padding for images transformed with warp_affine ops
        (translation, shear and rotate). If `None` is specified, the images are padded
        with the border value repeated (clamped).
    interp_type: types.DALIInterpType, optional
        Interpolation method used by the warp_affine ops (translation, shear and rotate).
        Supported values are `types.INTERP_LINEAR` (default) and `types.INTERP_NN`.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    aug_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    use_shape = shape is not None
    if use_shape:
        aug_kwargs["shape"] = shape
    image_net_policy = get_image_net_policy(use_shape=use_shape,
                                            max_translate_abs=max_translate_abs,
                                            max_translate_rel=max_translate_rel)
    return apply_auto_augment(image_net_policy, sample, seed, **aug_kwargs)


def apply_auto_augment(policy: Policy, sample: _DataNode, seed: Optional[int] = None,
                       **kwargs) -> _DataNode:
    """
    Applies AutoAugment (https://arxiv.org/abs/1805.09501) augmentation scheme to the
    provided batch of samples.

    Parameter
    ---------
    policy: Policy
        Set of sequences of augmentations to be applied in AutoAugment fashion.
    sample : DataNode
        A batch of samples to be processed.
    seed: int, optional
        Seed to be used to randomly sample operations (and to negate magnitudes).
    kwargs:
        A dictionary of extra parameters to be passed when calling augmentations.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        random augment suite accept `shapes`, `fill_value` and `interp_type`.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if len(policy.sub_policies) == 0:
        raise Exception(f"Cannot run empty policy. Got {policy} in `apply_auto_augment` call.")
    max_policy_len = max(len(sub_policy) for sub_policy in policy.sub_policies)
    should_run = fn.random.uniform(range=[0, 1], shape=(max_policy_len, ), dtype=types.FLOAT)
    sub_policy_id = fn.random.uniform(values=list(range(len(policy.sub_policies))), seed=seed,
                                      dtype=types.INT32)
    run_probabilities = _sub_policy_to_probability_map(policy)[sub_policy_id]
    magnitude_bins = _sub_policy_to_magnitude_bin_map(policy)[sub_policy_id]
    aug_ids, augmentations = _sub_policy_to_augmentation_map(policy)
    aug_ids = aug_ids[sub_policy_id]
    use_signed_magnitudes = any(aug.randomly_negate for aug in policy.augmentations.values())
    _forbid_unused_kwargs(augmentations, kwargs, 'apply_auto_augment')
    for stage_id in range(max_policy_len):
        magnitude_bin = magnitude_bins[stage_id]
        if use_signed_magnitudes:
            magnitude_bin = signed_bin(magnitude_bin)
        if should_run[stage_id] < run_probabilities[stage_id]:
            op_kwargs = dict(sample=sample, magnitude_bin=magnitude_bin,
                             num_magnitude_bins=policy.num_magnitude_bins, **kwargs)
            sample = _pretty_select(augmentations, aug_ids[stage_id], op_kwargs,
                                    auto_aug_name='apply_auto_augment',
                                    ref_suite_name='get_image_net_policy')
    return sample


def get_image_net_policy(use_shape: bool = False, max_translate_abs: Optional[int] = None,
                         max_translate_rel: Optional[float] = None) -> Policy:
    """
    Creates augmentation policy tuned for the ImageNet as described in AutoAugment
    (https://arxiv.org/abs/1805.09501).
    The returned policy can be run with `apply_auto_augment`.

    Parameter
    ---------
    use_shape : bool
        If true, the translation offset is computed as a percentage of the image. Useful if the
        images processed with the auto augment have different shapes. If false, the offsets range
        is bounded by a constant (`max_translate_abs`).
    max_translate_abs: int or (int, int), optional
        Only valid with use_shape=False, specifies the maximal shift (in pixels) in the translation
        augmentations. If tuple is specified, the first component limits height, the second the
        width.
    max_translate_rel: float or (float, float), optional
        Only valid with use_shape=True, specifies the maximal shift as a fraction of image shape
        in the translation augmentations. If tuple is specified, the first component limits
        height, the second the width.
    """
    translate_y = _get_translate_y(use_shape, max_translate_abs, max_translate_rel)
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
        name="ImageNetPolicy", num_magnitude_bins=11, sub_policies=[
            [(equalize, 0.8, 1), (shear_y, 0.8, 4)],
            [(color, 0.4, 9), (equalize, 0.6, 3)],
            [(color, 0.4, 1), (rotate, 0.6, 8)],
            [(solarize, 0.8, 3), (equalize, 0.4, 7)],
            [(solarize, 0.4, 2), (solarize, 0.6, 2)],
            [(color, 0.2, 0), (equalize, 0.8, 8)],
            [(equalize, 0.4, 8), (solarize_add, 0.8, 3)],
            [(shear_x, 0.2, 9), (rotate, 0.6, 8)],
            [(color, 0.6, 1), (equalize, 1.0, 2)],
            [(invert, 0.4, 9), (rotate, 0.6, 0)],
            [(equalize, 1.0, 9), (shear_y, 0.6, 3)],
            [(color, 0.4, 7), (equalize, 0.6, 0)],
            [(posterize, 0.4, 6), (auto_contrast, 0.4, 7)],
            [(solarize, 0.6, 8), (color, 0.6, 9)],
            [(solarize, 0.2, 4), (rotate, 0.8, 9)],
            [(rotate, 1.0, 7), (translate_y, 0.8, 9)],
            [(shear_x, 0.0, 0), (solarize, 0.8, 4)],
            [(shear_y, 0.8, 0), (color, 0.6, 4)],
            [(color, 1.0, 0), (rotate, 0.6, 2)],
            [(equalize, 0.8, 4)],
            [(equalize, 1.0, 4), (auto_contrast, 0.6, 2)],
            [(shear_y, 0.4, 7), (solarize_add, 0.6, 7)],
            [(posterize, 0.8, 2), (solarize, 0.6, 10)],
            [(solarize, 0.6, 8), (equalize, 0.6, 1)],
            [(color, 0.8, 6), (rotate, 0.4, 5)],
        ])


def _get_translate_y(use_shape: bool = False, max_translate_abs: Optional[int] = None,
                     max_translate_rel: Optional[float] = None):
    max_translate_height, _ = _parse_validate_offset(use_shape, max_translate_abs=max_translate_abs,
                                                     max_translate_rel=max_translate_rel,
                                                     default_translate_abs=250,
                                                     default_translate_rel=1.)
    if use_shape:
        return a.translate_y.augmentation((0, max_translate_height), True)
    else:
        return a.translate_y_no_shape.augmentation((0, max_translate_height), True)


def _sub_policy_to_probability_map(policy: Policy) -> _DataNode:
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    prob = np.array([[0. for _ in range(max_policy_len)] for _ in range(len(sub_policies))],
                    dtype=np.float32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            prob[sub_policy_id, stage_idx] = p
    return types.Constant(prob)


def _sub_policy_to_magnitude_bin_map(policy: Policy) -> _DataNode:
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    magnitude_bin = np.array([[0 for _ in range(max_policy_len)] for _ in range(len(sub_policies))],
                             dtype=np.int32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            magnitude_bin[sub_policy_id, stage_idx] = mag
    return types.Constant(magnitude_bin)


def _sub_policy_to_augmentation_map(policy: Policy) -> Tuple[_DataNode, List[_Augmentation]]:
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    augmentations = list(policy.augmentations.values()) + [a.identity]
    identity_id = len(augmentations) - 1
    augment_to_id = {augmentation: i for i, augmentation in enumerate(augmentations)}
    augments_by_id = np.array([[identity_id for _ in range(max_policy_len)]
                               for _ in range(len(sub_policies))], dtype=np.int32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (augment, p, mag) in enumerate(sub_policy):
            augments_by_id[sub_policy_id, stage_idx] = augment_to_id[augment]
    return types.Constant(augments_by_id), augmentations
