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

from types import MappingProxyType

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug import augmentations as a
from nvidia.dali.auto_aug.core.utils import select

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


class Policy:

    def __init__(self, name, num_magnitude_bins, augmentations, sub_policies):
        self.name = name
        self.num_magnitude_bins = num_magnitude_bins
        # prevent accidental modifications
        self.augmentations = MappingProxyType(augmentations)
        self.sub_policies = tuple(sub_policies)

    def __repr__(self):
        return (f"Policy({self.name}, {self.num_magnitude_bins}, "
                f"{self.augmentations}, {self.sub_policies})")


auto_augment_image_net_policy = Policy(
    "ImageNet", 11, {
        "shear_x": a.shear_x.augmentation((0, 0.3), True),
        "shear_y": a.shear_y.augmentation((0, 0.3), True),
        "translate_x": a.translate_x.augmentation((0, 0.45), True),
        "translate_y": a.translate_y.augmentation((0, 0.45), True),
        "rotate": a.rotate.augmentation((0, 30), True),
        "brightness": a.brightness.augmentation((0.1, 1.9), False, None),
        "contrast": a.contrast.augmentation((0.1, 1.9), False, None),
        "color": a.color.augmentation((0.1, 1.9), False, None),
        "sharpness": a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
        "posterize": a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
        "solarize": a.solarize.augmentation((0, 256), False),
        "solarize_add": a.solarize_add.augmentation((0, 110), False),
        "invert": a.invert,
        "equalize": a.equalize,
        "auto_contrast": a.auto_contrast,
    }, [
        [("equalize", 0.8, 1), ('shear_y', 0.8, 4)],
        [('color', 0.4, 9), ('equalize', 0.6, 3)],
        [('color', 0.4, 1), ('rotate', 0.6, 8)],
        [('solarize', 0.8, 3), ('equalize', 0.4, 7)],
        [('solarize', 0.4, 2), ('solarize', 0.6, 2)],
        [('color', 0.2, 0), ('equalize', 0.8, 8)],
        [('equalize', 0.4, 8), ('solarize_add', 0.8, 3)],
        [('shear_x', 0.2, 9), ('rotate', 0.6, 8)],
        [('color', 0.6, 1), ('equalize', 1.0, 2)],
        [('invert', 0.4, 9), ('rotate', 0.6, 0)],
        [('equalize', 1.0, 9), ('shear_y', 0.6, 3)],
        [('color', 0.4, 7), ('equalize', 0.6, 0)],
        [('posterize', 0.4, 6), ('auto_contrast', 0.4, 7)],
        [('solarize', 0.6, 8), ('color', 0.6, 9)],
        [('solarize', 0.2, 4), ('rotate', 0.8, 9)],
        [('rotate', 1.0, 7), ('translate_y', 0.8, 9)],
        [('shear_x', 0.0, 0), ('solarize', 0.8, 4)],
        [('shear_y', 0.8, 0), ('color', 0.6, 4)],
        [('color', 1.0, 0), ('rotate', 0.6, 2)],
        [('equalize', 0.8, 4)],
        [('equalize', 1.0, 4), ('auto_contrast', 0.6, 2)],
        [('shear_y', 0.4, 7), ('solarize_add', 0.6, 7)],
        [('posterize', 0.8, 2), ('solarize', 0.6, 10)],
        [('solarize', 0.6, 8), ('equalize', 0.6, 1)],
        [('color', 0.8, 6), ('rotate', 0.4, 5)],
    ])


def auto_augment_image_net(samples, shapes=None, fill_value=0, interp_type=None,
                           max_translate_height=250, max_translate_width=250, seed=None):
    """
    Applies `auto_augment_image_net_policy` in AutoAugment (https://arxiv.org/abs/1805.09501)
    fashion to the provided batch of samples.

    Parameter
    ---------
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    shapes: DataNode, optional
        A batch of shapes of the `samples`. If specified, the `translation` operations
        are applied relative to the shape of the sample. Otherwise `max_translate_width`
        and `max_translate_height` constants are used to compute the magnitude of the
        translation.
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
    kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    name = auto_augment_image_net_policy.name
    num_magnitude_bins = auto_augment_image_net_policy.num_magnitude_bins
    augments = dict(auto_augment_image_net_policy.augmentations)
    sub_policies = auto_augment_image_net_policy.sub_policies
    if shapes is not None:
        augments["translate_x"] = a.translate_x_no_shape.augmentation((0, max_translate_width))
        augments["translate_y"] = a.translate_y_no_shape.augmentation((0, max_translate_height))
        kwargs["shapes"] = shapes
    policy = Policy(name, num_magnitude_bins, augments, sub_policies)
    return apply_auto_augment(policy, samples, seed, **kwargs)


def apply_auto_augment(policy: Policy, samples, seed=None, **kwargs):
    """
    Applies AutoAugment (https://arxiv.org/abs/1805.09501) augmentation scheme to the
    provided batch of samples.

    Parameter
    ---------
    policy: Policy
        Set of sequences of augmentations to be applied in AutoAugment fashion.
    samples : DataNode
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
        return samples
    max_policy_len = max(len(sub_policy) for sub_policy in policy.sub_policies)
    use_signed_magnitudes = any(aug.randomly_negate for aug in policy.augmentations.values())
    if not use_signed_magnitudes:
        random_sign = None
    else:
        random_sign = fn.random.uniform(values=[0, 1], seed=seed, shape=(max_policy_len, ),
                                        dtype=types.INT32)
    should_run = fn.random.uniform(range=[0, 1], shape=(max_policy_len, ), dtype=types.FLOAT)
    policy_id = fn.random.uniform(values=list(range(len(policy.sub_policies))), seed=seed,
                                  dtype=types.INT32)
    run_probabilities = get_probabilities(policy)
    run_probabilities = run_probabilities[policy_id]
    magnitudes = get_magnitudes(policy)
    magnitudes = magnitudes[policy_id]
    aug_ids, augmentations = get_augmentations(policy)
    aug_ids = aug_ids[policy_id]
    for stage_id in range(max_policy_len):
        if should_run[stage_id] < run_probabilities[stage_id]:
            op_kwargs = dict(samples=samples, magnitude_bin_idx=magnitudes[stage_id],
                             num_magnitude_bins=policy.num_magnitude_bins,
                             random_sign=random_sign[stage_id], **kwargs)
            samples = select(augmentations, aug_ids[stage_id], op_kwargs)
    return samples


def get_probabilities(policy: Policy):
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    prob = np.array([[0. for _ in range(max_policy_len)] for _ in range(len(sub_policies))],
                    dtype=np.float32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            prob[sub_policy_id, stage_idx] = p
    return types.Constant(prob)


def get_augmentations(policy: Policy):
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    unique_op_names = list(
        set(op_name for sub_policy in sub_policies for op_name, p, mag in sub_policy))
    num_ops = len(unique_op_names)
    op_name_to_id_map = {name: i for i, name in enumerate(unique_op_names)}
    identity_id = num_ops
    op_ids = np.array([[identity_id for _ in range(max_policy_len)]
                       for _ in range(len(sub_policies))], dtype=np.int32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (op_name, p, mag) in enumerate(sub_policy):
            op_ids[sub_policy_id, stage_idx] = op_name_to_id_map[op_name]
    ops = [policy.augmentations[op_name] for op_name in unique_op_names] + [a.identity]
    return types.Constant(op_ids), ops


def get_magnitudes(policy: Policy):
    sub_policies = policy.sub_policies
    max_policy_len = max(len(sub_policy) for sub_policy in sub_policies)
    magnitude_bin = np.array([[0 for _ in range(max_policy_len)] for _ in range(len(sub_policies))],
                             dtype=np.int32)
    for sub_policy_id, sub_policy in enumerate(sub_policies):
        for stage_idx, (aug_name, p, mag) in enumerate(sub_policy):
            magnitude_bin[sub_policy_id, stage_idx] = mag
    return types.Constant(magnitude_bin)
