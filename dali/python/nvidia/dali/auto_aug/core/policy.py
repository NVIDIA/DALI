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

from nvidia.dali.auto_aug.core._augmentation import Augmentation
from typing import Optional, Sequence, Tuple


class Policy:
    def __init__(
        self,
        name: str,
        num_magnitude_bins: int,
        sub_policies: Sequence[Sequence[Tuple[Augmentation, float, Optional[int]]]],
    ):
        """
        Describes the augmentation policy as introduced in AutoAugment
        (https://arxiv.org/abs/1805.09501).

        Args
        ----
        name : str
            A name of the policy, for presentation purposes.
        num_magnitude_bins : int
            The number of bins that augmentations' magnitude ranges should be divided into.
        sub_policies: Sequence[Sequence[Tuple[Augmentation, float, Optional[int]]]]
            A list of sequences of transformations. For each processed sample, one of the
            sequences is chosen uniformly at random. Then, the tuples from the sequence
            are considered one by one. Each tuple describes what augmentation to apply at
            that point, what is the probability of skipping the augmentation at that time
            and what magnitude to use with the augmentation.
        """
        self.name = name
        self.num_magnitude_bins = num_magnitude_bins
        if not isinstance(num_magnitude_bins, int) or num_magnitude_bins < 1:
            raise Exception(
                f"The `num_magnitude_bins` must be a positive integer, got {num_magnitude_bins}."
            )
        if not isinstance(sub_policies, (list, tuple)):
            raise Exception(
                f"The `sub_policies` must be a list or tuple of sub policies, "
                f"got {type(sub_policies)}."
            )
        for sub_policy in sub_policies:
            if not isinstance(sub_policy, (list, tuple)):
                raise Exception(f"Each sub policy must be a list or tuple, got {sub_policy}.")
            for op_desc in sub_policy:
                if not isinstance(op_desc, (list, tuple)) or len(op_desc) != 3:
                    raise Exception(
                        f"Each operation in sub policy must be specified as a triple: "
                        f"(augmentation, probability, magnitude). Got {op_desc}."
                    )
                aug, p, mag = op_desc
                if not isinstance(aug, Augmentation):
                    raise Exception(
                        f"Each augmentation in sub policies must be an instance of "
                        f"Augmentation. Got `{aug}`. Did you forget to use `@augmentation` "
                        f"decorator?"
                    )
                if not isinstance(p, (float, int)) or not 0 <= p <= 1:
                    raise Exception(
                        f"Probability of applying the augmentation must be a number from "
                        f"`[0, 1]` range. Got `{p}` for augmentation `{aug.name}`."
                    )
                if p == 0:
                    warnings.warn(
                        f"The augmentation `{aug.name}` in policy `{name}` is used with "
                        f"probability 0 in one of the sub-policies."
                    )
                if mag is None:
                    if aug.mag_range is not None:
                        raise Exception(
                            f"The augmentation `{aug.name}` has `mag_range` specified, so the "
                            f"magnitude bin is required. However, got `None` in the policy "
                            f"`{name}`."
                        )
                else:
                    if aug.mag_range is None:
                        warnings.warn(
                            f"The magnitude bin `{mag}` for augmentation `{aug.name}` in policy "
                            f"`{name}` will be ignored. The augmentation does not accept "
                            f"magnitudes (as it has no `mag_range` specified). You can specify "
                            f"`None` instead of `{mag}` to silence this warning."
                        )
                    if not isinstance(mag, int) or not 0 <= mag < self.num_magnitude_bins:
                        raise Exception(
                            f"Magnitude of the augmentation must be an integer from "
                            f"`[0, {num_magnitude_bins - 1}]` range. "
                            f"Got `{mag}` for augmentation `{aug.name}`."
                        )
        self.sub_policies = _sub_policy_with_unique_names(sub_policies)

    @property
    def augmentations(self):
        augments = set(aug for sub_policy in self.sub_policies for aug, p, mag in sub_policy)
        augments = sorted(list(augments), key=lambda aug: aug.name)
        return {augment.name: augment for augment in augments}

    def __repr__(self):
        sub_policies_repr = ",\n\t".join(
            repr([(augment.name, p, mag) for augment, p, mag in sub_policy])
            for sub_policy in self.sub_policies
        )
        sub_policies_repr_sep = "" if not sub_policies_repr else "\n\t"
        augmentations_repr = ",\n\t".join(
            f"'{name}': {repr(augment)}" for name, augment in self.augmentations.items()
        )
        augmentations_repr_sep = "" if not augmentations_repr else "\n\t"
        return (
            f"Policy(name={repr(self.name)}, num_magnitude_bins={repr(self.num_magnitude_bins)}, "
            f"sub_policies=[{sub_policies_repr_sep}{sub_policies_repr}], "
            f"augmentations={{{augmentations_repr_sep}{augmentations_repr}}})"
        )


def _sub_policy_with_unique_names(
    sub_policies: Sequence[Sequence[Tuple[Augmentation, float, Optional[int]]]],
) -> Sequence[Sequence[Tuple[Augmentation, float, Optional[int]]]]:
    """
    Check if the augmentations used in the sub-policies have unique names.
    If not, rename them by adding enumeration to the names.
    The aim is to have non-ambiguous presentation.
    """
    all_augments = [aug for sub_policy in sub_policies for aug, p, mag in sub_policy]
    augments = set(all_augments)
    names = set(aug.name for aug in augments)
    if len(names) == len(augments):
        return tuple(tuple(sub_policy) for sub_policy in sub_policies)
    num_digits = len(str(len(augments) - 1))
    remap_aug = {}
    i = 0
    for augment in all_augments:
        if augment not in remap_aug:
            remap_aug[augment] = augment.augmentation(
                name=f"{str(i).zfill(num_digits)}__{augment.name}"
            )
            i += 1
    return tuple(
        tuple((remap_aug[aug], p, mag) for aug, p, mag in sub_policy) for sub_policy in sub_policies
    )
