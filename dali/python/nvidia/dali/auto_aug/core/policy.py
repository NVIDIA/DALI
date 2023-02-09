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

from nvidia.dali.auto_aug.core._augmentation import Augmentation
from typing import Sequence, Tuple


class Policy:

    def __init__(self, name: str, num_magnitude_bins: int,
                 sub_policies: Sequence[Sequence[Tuple[Augmentation, float, int]]]):
        """
        Describes the augmentation policy as introduced in AutoAugment
        (https://arxiv.org/abs/1805.09501).

        Parameter
        ---------
        name : str
            A name of the policy, for presentation purposes.
        num_magnitude_bins : int
            The number of bins that augmentations' magnitude ranges should be divided into.
        sub_policies: Sequence[Sequence[Tuple[Augmentation, float, int]]]
            A list of sequences of transformations. For each processed sample, one of the
            sequences is chosen uniformly at random. Then, the tuples from the sequence
            are considered one by one. Each tuple describes what augmentation to apply at
            that point, what is the probability of skipping the augmentation at that time
            and what magnitude to use with the augmentation.
        """
        self.name = name
        self.num_magnitude_bins = num_magnitude_bins
        if not isinstance(sub_policies, (list, tuple)):
            raise Exception(f"The `sub_policies` must be a list or tuple of sub policies, "
                            f"got {type(sub_policies)}.")
        for sub_policy in sub_policies:
            if not isinstance(sub_policy, (list, tuple)):
                raise Exception(f"Each sub policy must be a list or tuple, got {sub_policy}.")
            for op_desc in sub_policy:
                if not isinstance(op_desc, (list, tuple)) or len(op_desc) != 3:
                    raise Exception(f"Each operation in sub policy must be specified as a triple: "
                                    f"(augmentation, probability, magnitude). Got {op_desc}.")
                if not isinstance(op_desc[0], Augmentation):
                    raise Exception(
                        f"Each augmentation in sub policies must be an instance of "
                        f"Augmentation. Got {op_desc[0]}. Did you forget to use `@augmentation` "
                        f"decorator?")
        self.sub_policies = sub_policy_with_unique_names(sub_policies)

    @property
    def augmentations(self):
        augments = set(aug for sub_policy in self.sub_policies for aug, p, mag in sub_policy)
        augments = sorted(list(augments), key=lambda aug: aug.name)
        return {augment.name: augment for augment in augments}

    def __repr__(self):
        sub_policies_repr = ",\n\t".join(
            repr([(augment.name, p, mag) for augment, p, mag in sub_policy])
            for sub_policy in self.sub_policies)
        augmentations_repr = ",\n\t".join(f"'{name}': {repr(augment)}"
                                          for name, augment in self.augmentations.items())
        return (
            f"Policy(name={repr(self.name)}, num_magnitude_bins={repr(self.num_magnitude_bins)}, "
            f"sub_policies=[\n\t{sub_policies_repr}], augmentations={{\n\t{augmentations_repr}}})")


def sub_policy_with_unique_names(
    sub_policies: Sequence[Sequence[Tuple[Augmentation, float, int]]]
) -> Tuple[Tuple[Tuple[Augmentation, float, int]]]:
    augments = set(aug for sub_policy in sub_policies for aug, p, mag in sub_policy)
    names = set(aug.name for aug in augments)
    if len(names) == len(augments):
        return tuple(tuple(sub_policy) for sub_policy in sub_policies)
    aug_by_name = {name: [] for name in names}
    for aug in augments:
        aug_by_name[aug.name].append(aug)
    remap_aug = {}
    for aug_name, augs in aug_by_name.items():
        if len(augs) == 1:
            [aug] = augs
            remap_aug[aug] = aug
        else:
            for i, aug in enumerate(augs):
                remap_aug[aug] = aug.augmentation(name=f"{aug_name}__{i}")
    return tuple(
        tuple((remap_aug[aug], p, mag) for aug, p, mag in sub_policy)
        for sub_policy in sub_policies)
