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
from nvidia.dali.auto_aug.core.utils import operation_idx_random_choice, select

rand_augment_suite = {
    "shear_x": a.shear_x.augmentation((0, 0.3), True),
    "shear_y": a.shear_y.augmentation((0, 0.3), True),
    "translate_x": a.translate_x.augmentation((0, 0.45), True),
    "translate_y": a.translate_y.augmentation((0, 0.45), True),
    "rotate": a.rotate.augmentation((0, 30), True),
    "brightness": a.brightness.augmentation((0, 0.9), True, a.shift_enhance_range),
    "contrast": a.contrast.augmentation((0, 0.9), True, a.shift_enhance_range),
    "color": a.color.augmentation((0, 0.9), True, a.shift_enhance_range),
    "sharpness": a.sharpness.augmentation((0, 0.9), True, a.sharpness_kernel),
    "posterize": a.posterize.augmentation((8, 4), False, a.poster_mask_uint8),
    # solarization strength increases with decreasing magnitude (threshold)
    "solarize": a.solarize.augmentation((256, 0)),
    "equalize": a.equalize,
    "auto_contrast": a.auto_contrast,
    "identity": a.identity,
}

non_monotonic_augs = {
    "posterize": a.posterize.augmentation((0, 4), False, a.poster_mask_uint8),
    "solarize": a.solarize.augmentation((0, 256), False, None),
    "brightness": a.brightness.augmentation((0.1, 1.9), False, None),
    "contrast": a.contrast.augmentation((0.1, 1.9), False, None),
    "color": a.color.augmentation((0.1, 1.9), False, None),
    "sharpness": a.sharpness.augmentation((0.1, 1.9), False, a.sharpness_kernel_shifted),
}


def rand_augment(samples, n, m, num_magnitude_bins=31, shapes=None, fill_value=0, interp_type=None,
                 max_translate_height=250, max_translate_width=250, seed=None, monotonic_mag=True,
                 excluded=None):
    """
    Applies RandAugment (https://arxiv.org/abs/1909.13719) augmentation scheme to the
    provided batch of samples.

    Parameter
    ---------
    samples : DataNode
        A batch of samples to be processed. The samples should be images of `HWC` layout,
        `uint8` type and reside on GPU.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude (strength) of each operation to be applied, it must be an integer
        within `[0, num_magnitude_bins - 1]`.
    num_magnitude_bins: int, optional
        The number of bins to divide the magnitude ranges into.
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
    monotonic_mag: bool, optional
        There are two flavours of RandAugment available in different frameworks. For the default
        `monotonic_mag=True` the strength of operations that accept magnitude increases with
        the increasing magnitudes. If set to False, magnitudes for some color operations differ.
        There, the `posterize` and `solarize` strength decreases with increasing magnitudes and
        enhance operations (`brightness`, `contrast`, `color`, `sharpness`) use (0.1, 1.9) range,
        which means that the strength decreases the closer the magnitudes are to the center
        of the range. The affected ops are listed in `non_monotonic_augs`.
    excluded: List[str], optional
        A list of names of the operations to be excluded from the `rand_augment_suite`.
        If, instead of just limiting the set of operations, you need to include some custom
        operations or fine-tune the existing ones, you can use the `apply_rand_augment`
        directly, which accepts a list of augmentations.

    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    augments = dict(**rand_augment_suite)
    augment_kwargs = {"fill_value": fill_value, "interp_type": interp_type}
    if shapes is not None:
        augment_kwargs["shapes"] = shapes
    else:
        augments["translate_x"] = a.translate_x_no_shape.augmentation((0, max_translate_width))
        augments["translate_y"] = a.translate_y_no_shape.augmentation((0, max_translate_height))
    if not monotonic_mag:
        augments.update(non_monotonic_augs)
    excluded = excluded or tuple()
    for name in excluded:
        if name not in augments:
            raise Exception(f"The `{name}` was specified in `excluded`, but the rand_augment_suite "
                            f"does not contain such an augmentation.")
    selected_augments = [aug for name, aug in augments.items() if name not in excluded]
    return apply_rand_augment(selected_augments, samples, n, m,
                              num_magnitude_bins=num_magnitude_bins, seed=seed,
                              augment_kwargs=augment_kwargs)


def apply_rand_augment(augmentations, samples, n, m, num_magnitude_bins, seed, augment_kwargs=None):
    """
    Applies RandAugment (https://arxiv.org/abs/1909.13719) like transformations but with custom
    set of augmentations.

    Parameter
    ---------
    augmentations : List[core.Augmentation]
        List of augmentations to be sampled and applied in RandAugment fashion.
    samples : DataNode
        A batch of samples to be processed.
    n: int
        The number of randomly sampled operations to be applied to a sample.
    m: int
        A magnitude (strength) of each operation to be applied, it must be an integer
        within `[0, num_magnitude_bins - 1]`.
    num_magnitude_bins: int
        The number of bins to divide the magnitude ranges into.
    seed: int
        Seed to be used to randomly sample operations (and to negate magnitudes).
    augment_kwargs:
        A dictionary of extra parameters to be passed when calling `augmentations`.
        The signature of each augmentation is checked for any extra arguments and if
        the name of the argument matches one from the `augment_kwargs`, the value is
        passed as an argument. For example, some augmentations from the default
        random augment suite accept `shapes`, `fill_value` and `interp_type`.
    Returns
    -------
    DataNode
        A batch of transformed samples.
    """
    if len(augmentations) == 0:
        return samples
    augment_kwargs = augment_kwargs or {}
    use_signed_magnitudes = any(aug.randomly_negate for aug in augmentations)
    if not use_signed_magnitudes:
        random_sign = None
    else:
        random_sign = fn.random.uniform(range=[0, 1], dtype=types.INT32, seed=seed,
                                        shape=tuple() if n == 1 else (n, ))
    op_idx = operation_idx_random_choice(len(augmentations), n, seed)
    for level_idx in range(n):
        if not use_signed_magnitudes or n == 1:
            level_random_sign = random_sign
        else:
            level_random_sign = random_sign[level_idx]
        op_kwargs = dict(samples=samples, magnitude_bin_idx=m,
                         num_magnitude_bins=num_magnitude_bins, random_sign=level_random_sign,
                         **augment_kwargs)
        level_op_idx = op_idx if n == 1 else op_idx[level_idx]
        samples = select(augmentations, level_op_idx, op_kwargs)
    return samples
