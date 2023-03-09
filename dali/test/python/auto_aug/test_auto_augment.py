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

import itertools
import os

import numpy as np
from nose2.tools import params

from nvidia.dali import fn, types
from nvidia.dali.pipeline import experimental
from nvidia.dali.auto_aug import auto_augment, augmentations as a
from nvidia.dali.auto_aug.core import augmentation, Policy

from test_utils import get_dali_extra_path
from nose_utils import assert_raises

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


@params(*tuple(enumerate(itertools.product((True, False), (True, False), (None, 0),
                                           (True, False)))))
def test_run_auto_aug(i, args):
    uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 64, 128]
    batch_size = batch_sizes[i % len(batch_sizes)]

    @experimental.pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4,
                               device_id=0, seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        if uniformly_resized:
            image = fn.resize(image, size=(244, 244))
        extra = {} if not use_shape else {"shape": fn.peek_image_shape(encoded_image)}
        if fill_value is not None:
            extra["fill_value"] = fill_value
        if specify_translation_bounds:
            if use_shape:
                extra["max_translate_rel"] = 0.9
            else:
                extra["max_translate_abs"] = 400
        image = auto_augment.auto_augment_image_net(image, **extra)
        return image

    p = pipeline()
    p.build()
    for _ in range(3):
        p.run()


@params((True, 0), (True, 1), (False, 0), (False, 1))
def test_translation(use_shape, offset_fraction):
    # make sure the translation helper processes the args properly
    max_extent = 1000
    fill_value = 217
    params = {}
    if use_shape:
        params["max_translate_rel"] = offset_fraction
    else:
        params["max_translate_abs"] = offset_fraction * max_extent
    translate_y = auto_augment._get_translate_y(use_shape=use_shape, **params)
    policy = Policy(f"Policy_{use_shape}_{offset_fraction}", num_magnitude_bins=21,
                    sub_policies=[[(translate_y, 1, 20)]])

    @experimental.pipeline_def(enable_conditionals=True, batch_size=16, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image = fn.crop(image, crop=(max_extent, max_extent), out_of_bounds_policy="trim_to_shape")
        if use_shape:
            return auto_augment.apply_auto_augment(policy, image, fill_value=fill_value,
                                                   shape=fn.peek_image_shape(encoded_image))
        else:
            return auto_augment.apply_auto_augment(policy, image, fill_value=fill_value)

    p = pipeline()
    p.build()
    output, = p.run()
    output = [np.array(sample) for sample in output.as_cpu()]
    for i, sample in enumerate(output):
        sample = np.array(sample)
        if offset_fraction == 1:
            assert np.all(sample == fill_value), f"sample_idx: {i}"
        else:
            assert np.sum(sample == fill_value) / sample.size < 0.1, f"sample_idx: {i}"


@params(
    (False, "cpu", 256),
    (False, "gpu", 512),
    (True, "cpu", 128),
    (True, "gpu", 348),
)
def test_sub_policy(randomly_negate, dev, batch_size):

    num_magnitude_bins = 10

    def as_param_with_op_id(op_id):

        def as_param(magnitude):
            return np.array([op_id, magnitude], dtype=np.int32)

        return as_param

    @augmentation(
        mag_range=(0, 9),
        as_param=as_param_with_op_id(1),
        param_device=dev,
    )
    def first(sample, op_id_mag_id):
        return fn.cat(sample, op_id_mag_id)

    @augmentation(
        mag_range=(10, 19),
        as_param=as_param_with_op_id(2),
        randomly_negate=randomly_negate,
        param_device=dev,
    )
    def second(sample, op_id_mag_id):
        return fn.cat(sample, op_id_mag_id)

    @augmentation(
        mag_range=(20, 29),
        as_param=as_param_with_op_id(3),
        param_device=dev,
    )
    def third(sample, op_id_mag_id):
        return fn.cat(sample, op_id_mag_id)

    sub_policies = [
        [(first, 1, 0), (second, 1, 5), (third, 1, 3)],
        [(first, 1, 1), (third, 1, 4), (first, 1, 2)],
        [(second, 1, 2), (first, 1, 3), (third, 1, 4)],
        [(second, 1, 3), (third, 1, 2), (first, 1, 5)],
        [(third, 1, 4), (first, 1, 1), (second, 1, 1)],
        [(third, 1, 5), (second, 1, 9), (first, 1, 2)],
        [(first, 1, 6), (first, 1, 1)],
        [(third, 1, 7)],
        [(first, 1, 8), (first, 1, 4), (second, 1, 7), (second, 1, 6)],
    ]

    policy = Policy("MyPolicy", num_magnitude_bins=num_magnitude_bins, sub_policies=sub_policies)

    @experimental.pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4,
                               device_id=0, seed=44)
    def pipeline():
        sample = types.Constant(np.array([], dtype=np.int32), device=dev)
        if dev == "gpu":
            sample = sample.gpu()
        sample = auto_augment.apply_auto_augment(policy, sample)
        return fn.reshape(sample, shape=(-1, 2))

    p = pipeline()
    p.build()

    sub_policy_outputs = []
    for sub_policy in sub_policies:
        out = []
        for aug, _, mag_bin in sub_policy:
            magnitudes = aug._get_magnitudes(num_magnitude_bins)
            param = aug._map_mag_to_param(magnitudes[mag_bin])
            out.append(param)
        sub_policy_outputs.append(out)

    # magnitudes are chosen so that the magnitude of the first op in
    # each sub-policy identifies the sub-policy
    output_cases = {out[0][1]: np.array(out) for out in sub_policy_outputs}

    for _ in range(5):
        output, = p.run()
        if dev == "gpu":
            output = output.as_cpu()
        output = [np.array(sample) for sample in output]
        for sample in output:
            test_sample = sample if not randomly_negate else np.abs(sample)
            np.testing.assert_equal(np.abs(test_sample), output_cases[test_sample[0][1]])
        if randomly_negate:
            count = 0
            for sample in output:
                for op_mag in sample:
                    if op_mag[1] < 0:
                        # only the `second` augmentation is marked as randomly_negated
                        assert op_mag[0] == 2, f"{sample}"
                        count += 1
            # 1/9 (prob of particular policy) * 5 (sub policy with exactly one `second` aug)
            # * 1/2 (random sign) + 1/9 * 2 * 1/2 (last sub policy) = 5/18 + 2/18 = 7/18
            expected_negated = batch_size * 7 / 18
            eps = 0.1 * batch_size  # just a guess
            assert expected_negated - eps <= count <= expected_negated + eps, \
                f"{count} {expected_negated}"


def test_policy_presentation():

    empty_policy = Policy("EmptyPolicy", num_magnitude_bins=31, sub_policies=[])
    empty_policy_str = str(empty_policy)
    assert "sub_policies=[]" in empty_policy_str, empty_policy_str
    assert "augmentations={}" in empty_policy_str, empty_policy_str

    def get_first_augment():

        @augmentation
        def clashing_name(sample, _):
            return sample

        return clashing_name

    def get_second_augment():

        @augmentation
        def clashing_name(sample, _):
            return sample

        return clashing_name

    one = get_first_augment()
    another = get_second_augment()
    sub_policies = [[(one, 0.1, 5), (another, 0.4, 7)], [(another, 0.2, 1), (one, 0.5, 2)],
                    [(another, 0.7, 1)]]
    policy = Policy(name="DummyPolicy", num_magnitude_bins=11, sub_policies=sub_policies)
    assert policy.sub_policies[0][0][0] is policy.sub_policies[1][1][0]
    assert policy.sub_policies[0][1][0] is policy.sub_policies[1][0][0]
    assert policy.sub_policies[0][1][0] is policy.sub_policies[2][0][0]
    assert len(sub_policies) == len(policy.sub_policies)
    for sub_pol, pol_sub_pol in zip(sub_policies, policy.sub_policies):
        assert len(sub_pol) == len(pol_sub_pol)
        for (aug, p, mag), (pol_aug, pol_p, pol_mag) in zip(sub_pol, pol_sub_pol):
            assert p == pol_p, f"({aug}, {p}, {mag}), ({pol_aug}, {pol_p}, {pol_mag})"
            assert mag == pol_mag, f"({aug}, {p}, {mag}), ({pol_aug}, {pol_p}, {pol_mag})"

    @augmentation
    def yet_another_aug(sample, _):
        return sample

    sub_policies = [[(yet_another_aug, 0.5, i), (one.augmentation(mag_range=(0, i)), 0.24, i)]
                    for i in range(1, 107)]
    bigger_policy = Policy(name="BiggerPolicy", num_magnitude_bins=200, sub_policies=sub_policies)
    for i, (first, second) in enumerate(bigger_policy.sub_policies):
        assert first[0].name == '000__yet_another_aug', f"{second[0].name}"
        assert second[0].name == f'{(i + 1):03}__clashing_name', f"{second[0].name}"


def test_unused_arg_fail():

    @experimental.pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy()
        return auto_augment.apply_auto_augment(image_net_policy, image, misspelled_kwarg=100)

    msg = "The kwarg `misspelled_kwarg` is not used by any of the augmentations."
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_missing_shape_fail():

    @experimental.pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy(use_shape=True)
        return auto_augment.apply_auto_augment(image_net_policy, image)

    msg = "`translate_y` * provide it as `shape` argument to `apply_auto_augment` call"
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_wrong_sub_policy_format_fail():

    with assert_raises(Exception,
                       glob="The `num_magnitude_bins` must be a positive integer, got 0"):
        Policy("ShouldFail", 0.25, a.rotate)

    with assert_raises(Exception,
                       glob="The `sub_policies` must be a list or tuple of sub policies"):
        Policy("ShouldFail", 9, a.rotate)

    with assert_raises(Exception, glob="Each sub policy must be a list or tuple"):
        Policy("ShouldFail", 9, [a.rotate])

    with assert_raises(
            Exception,
            glob="as a triple: (augmentation, probability, magnitude). Got Augmentation"):
        Policy("ShouldFail", 9, [(a.rotate, a.shear_x)])

    with assert_raises(Exception, glob="must be an instance of Augmentation. Got 0.5"):
        Policy("ShouldFail", 9, [[(0.5, a.rotate, 3)]])

    with assert_raises(Exception,
                       glob="Probability * must be a number from `[[]0, 1[]]` range. Got 2"):
        Policy("ShouldFail", 9, [[(a.rotate, 2, 2)]])

    with assert_raises(Exception, glob="Magnitude ** `[[]0, 8[]]` range. Got -1"):
        Policy("ShouldFail", 9, [[(a.rotate, 1, -1)]])
