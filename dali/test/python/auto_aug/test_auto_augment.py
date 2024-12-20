# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
import random

import numpy as np
from scipy.stats import chisquare
from nose2.tools import params

from nvidia.dali import fn, types
from nvidia.dali import pipeline_def
from nvidia.dali.auto_aug import auto_augment, augmentations as a
from nvidia.dali.auto_aug.core import augmentation, Policy

from test_utils import get_dali_extra_path, check_batch
from nose_utils import assert_raises, assert_warns

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")
vid_dir = os.path.join(data_root, "db", "video", "sintel", "video_files")
vid_files = ["sintel_trailer-720p_2.mp4"]
vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]


def mag_to_param_with_op_id(op_id):
    def mag_to_param(magnitude):
        return np.array([op_id, magnitude], dtype=np.int32)

    return mag_to_param


@pipeline_def(enable_conditionals=True, num_threads=4, device_id=0, seed=44)
def concat_aug_pipeline(dev, policy):
    data = types.Constant(np.array([], dtype=np.int32), device=dev)
    if dev == "gpu":
        data = data.gpu()
    data = auto_augment.apply_auto_augment(policy, data)
    return fn.reshape(data, shape=(-1, 2))


def collect_sub_policy_outputs(sub_policies, num_magnitude_bins):
    sub_policy_outputs = []
    for sub_policy in sub_policies:
        out = []
        for aug, _, mag_bin in sub_policy:
            magnitudes = aug._get_magnitudes(num_magnitude_bins)
            param = aug._map_mag_to_param(magnitudes[mag_bin])
            out.append(param)
        sub_policy_outputs.append(out)
    return sub_policy_outputs


run_aug_shape = ("image_net", "reduced_cifar10", "svhn")
run_aug_shape_supporting_cases = (
    # reduce the number of test cases by running three predefine shape-aware policies in turns,
    ((run_aug_shape[i % 3],) + params)
    for i, params in enumerate(
        itertools.product(
            ("cpu", "gpu"),
            (True, False),
            (True, False),
            (None, 0),
            (True, False),
        )
    )
)

run_aug_no_translation_cases = itertools.product(
    ("reduced_image_net",),
    ("cpu", "gpu"),
    (True, False),
    (False,),
    (None, 0),
    (False,),
)


@params(
    *tuple(enumerate(itertools.chain(run_aug_shape_supporting_cases, run_aug_no_translation_cases)))
)
def test_run_auto_aug(i, args):
    policy_name, dev, uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 41]
    batch_size = batch_sizes[i % len(batch_sizes)]

    @pipeline_def(
        enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0, seed=43
    )
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="cpu" if dev == "cpu" else "mixed")
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
        image = auto_augment.auto_augment(image, policy_name, **extra)
        return image

    # run the pipeline twice to make sure instantiation preserves determinism
    p1 = pipeline()
    p2 = pipeline()
    for _ in range(3):
        (out1,) = p1.run()
        (out2,) = p2.run()
        check_batch(out1, out2)


class VideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        num_frames = 31
        roi_start = (90, 0)
        roi_end = (630, 1280)
        size_1 = (215, 128)
        size_2 = (215, 220)

        @pipeline_def(batch_size=6, device_id=0, num_threads=4, seed=42)
        def pipeline(size):
            video = fn.readers.video_resize(
                filenames=vid_filenames,
                sequence_length=num_frames,
                roi_start=roi_start,
                roi_end=roi_end,
                resize_x=size[1],
                resize_y=size[0],
                file_list_include_preceding_frame=True,
                device="gpu",
            )
            return video

        cls.vid_files = []
        for size in (size_1, size_2):
            p = pipeline(size=size)
            (out,) = p.run()
            cls.vid_files.extend(np.array(sample) for sample in out.as_cpu())

    @params(
        *tuple(
            enumerate(
                (
                    ("cpu", "image_net", 1, True),
                    ("cpu", "reduced_cifar10", 4, False),
                    ("cpu", "svhn", 17, True),
                    ("cpu", "reduced_image_net", 3, False),
                    ("gpu", "image_net", 21, False),
                    ("gpu", "reduced_cifar10", 3, True),
                    ("gpu", "svhn", 1, False),
                    ("gpu", "reduced_image_net", 5, False),
                )
            )
        )
    )
    def test_uniform(self, i, args):
        device, policy_name, batch_size, use_shape = args
        num_iterations = 3

        assert device in ("gpu", "cpu")

        @pipeline_def(
            batch_size=batch_size, device_id=0, num_threads=4, seed=205, enable_conditionals=True
        )
        def pipeline():
            rng = random.Random(42 + i)
            video = fn.external_source(
                source=lambda: list(rng.choices(self.vid_files, k=batch_size)),
                batch=True,
                layout="FHWC",
            )
            extra = {} if not use_shape else {"shape": video.shape()[1:]}
            if device == "gpu":
                video = video.gpu()
            video = auto_augment.auto_augment(video, policy_name, **extra)
            return video

        # run the pipeline twice to make sure instantiation preserves determinism
        p1 = pipeline()
        p2 = pipeline()

        for _ in range(num_iterations):
            (out1,) = p1.run()
            (out2,) = p2.run()
            check_batch(out1, out2)


@params(
    (False, "cpu", 256),
    (False, "gpu", 512),
    (True, "cpu", 2000),
    (True, "gpu", 2000),
)
def test_sub_policy(randomly_negate, dev, batch_size):
    num_magnitude_bins = 10

    @augmentation(
        mag_range=(0, 9),
        mag_to_param=mag_to_param_with_op_id(1),
        param_device=dev,
    )
    def first(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(10, 19),
        mag_to_param=mag_to_param_with_op_id(2),
        randomly_negate=randomly_negate,
        param_device=dev,
    )
    def second(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(20, 29),
        mag_to_param=mag_to_param_with_op_id(3),
        randomly_negate=randomly_negate,
        param_device=dev,
    )
    def third(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

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
    p = concat_aug_pipeline(batch_size=batch_size, dev=dev, policy=policy)

    sub_policy_outputs = collect_sub_policy_outputs(sub_policies, num_magnitude_bins)
    # magnitudes are chosen so that the magnitude of the first op in
    # each sub-policy identifies the sub-policy
    assert len({out[0][1] for out in sub_policy_outputs}) == len(sub_policy_outputs)
    output_cases = {out[0][1]: np.array(out) for out in sub_policy_outputs}

    sub_policy_negation_cases = []
    for sub_policy in sub_policies:
        negated = []
        for aug, _, _ in sub_policy:
            if aug.randomly_negate:
                negated.append((True, False))
            else:
                negated.append((False,))
        sub_policy_negation_cases.append(list(itertools.product(*negated)))
    assert len(sub_policy_outputs) == len(sub_policy_negation_cases)

    for _ in range(5):
        (output,) = p.run()
        if dev == "gpu":
            output = output.as_cpu()
        output = [np.array(sample) for sample in output]
        for sample in output:
            test_sample = sample if not randomly_negate else np.abs(sample)
            np.testing.assert_equal(np.abs(test_sample), output_cases[test_sample[0][1]])
            for op_mag in sample:
                if op_mag[1] < 0:
                    # the `second` and `third` augmentation are marked as randomly_negated
                    assert op_mag[0] in [2, 3], f"{sample}"
        if randomly_negate:
            # for each sub-policy, count occurrences of any possible sequence
            # of magnitude signs
            negation_cases = {
                out[0][1]: {case: 0 for case in cases}
                for out, cases in zip(sub_policy_outputs, sub_policy_negation_cases)
            }
            for sample in output:
                mag_signs = tuple(op_mag[1] < 0 for op_mag in sample)
                negation_cases[np.abs(sample[0][1])][mag_signs] += 1
            counts, expected_counts = [], []
            for sub_policy_cases in negation_cases.values():
                expected = batch_size / (len(sub_policies) * len(sub_policy_cases))
                for count in sub_policy_cases.values():
                    counts.append(count)
                    expected_counts.append(expected)
            stat = chisquare(counts, expected_counts)
            # assert that the magnitudes negation looks independently enough
            # (0.01 <=)
            assert 0.01 <= stat.pvalue, f"{stat}"


@params(("cpu",), ("gpu",))
def test_op_skipping(dev):
    num_magnitude_bins = 20
    batch_size = 2400

    @augmentation(
        mag_range=(0, 19),
        mag_to_param=mag_to_param_with_op_id(1),
        randomly_negate=True,
        param_device=dev,
    )
    def first(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(0, 19),
        mag_to_param=mag_to_param_with_op_id(2),
        randomly_negate=True,
        param_device=dev,
    )
    def second(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(0, 19),
        mag_to_param=mag_to_param_with_op_id(3),
        param_device=dev,
    )
    def third(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(0, 19),
        mag_to_param=mag_to_param_with_op_id(4),
        param_device=dev,
    )
    def first_stage_only(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    @augmentation(
        mag_range=(0, 19),
        mag_to_param=mag_to_param_with_op_id(5),
        param_device=dev,
    )
    def second_stage_only(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    sub_policies = [
        [(first, 0.5, 1), (first, 0.25, 2)],
        [(second, 0.8, 3), (second, 0.7, 4)],
        [(first, 0.9, 5), (second, 0.6, 6)],
        [(second, 0.3, 7), (first, 0.25, 8)],
        [(third, 1, 9), (third, 0.75, 10)],
        [(third, 0.3, 11), (first, 0.22, 12)],
        [(second, 0.6, 13), (third, 0, 14)],
        [(first_stage_only, 0.5, 15), (third, 0.7, 16)],
        [(third, 0.8, 17), (second_stage_only, 0.6, 18)],
    ]

    # sub_policy_cases = [[] for _ in range(len(sub_policies))]
    expected_counts = {tuple(): 0.0}
    for (left_aug, left_p, left_mag), (right_aug, right_p, right_mag) in sub_policies:
        expected_counts[tuple()] += (1.0 - left_p) * (1 - right_p) / len(sub_policies)
        only_left_p = left_p * (1 - right_p) / len(sub_policies)
        only_right_p = (1 - left_p) * right_p / len(sub_policies)
        for aug, mag, prob in [
            (left_aug, left_mag, only_left_p),
            (right_aug, right_mag, only_right_p),
        ]:
            if not aug.randomly_negate:
                expected_counts[(mag,)] = prob
            else:
                expected_counts[(mag,)] = prob / 2
                expected_counts[(-mag,)] = prob / 2
        sign_cases = [(-1, 1) if aug.randomly_negate else (1,) for aug in (left_aug, right_aug)]
        sign_cases = list(itertools.product(*sign_cases))
        prob = left_p * right_p / len(sub_policies)
        for left_sign, right_sign in sign_cases:
            mags = (left_sign * left_mag, right_sign * right_mag)
            expected_counts[mags] = prob / len(sign_cases)
    expected_counts = {mag: prob * batch_size for mag, prob in expected_counts.items() if prob > 0}
    assert all(num_elements >= 5 for num_elements in expected_counts.values()), (
        f"The batch size is too small (i.e. some output cases are expected less "
        f"than five times in the output): {expected_counts}"
    )

    policy = Policy("MyPolicy", num_magnitude_bins=num_magnitude_bins, sub_policies=sub_policies)
    p = concat_aug_pipeline(batch_size=batch_size, dev=dev, policy=policy, seed=1234)

    for _ in range(5):
        (output,) = p.run()
        if dev == "gpu":
            output = output.as_cpu()
        output = [np.array(sample) for sample in output]
        actual_counts = {allowed_case: 0 for allowed_case in expected_counts}
        for sample in output:
            mags = tuple(int(op_mag[1]) for op_mag in sample)
            actual_counts[mags] += 1

        actual, expected = [], []
        for mags in expected_counts:
            actual.append(actual_counts[mags])
            expected.append(expected_counts[mags])
        stat = chisquare(actual, expected)
        # assert that the magnitudes negation looks independently enough (0.01 <=)
        assert 0.01 <= stat.pvalue, f"{stat}"


def test_policy_presentation():
    empty_policy = Policy("EmptyPolicy", num_magnitude_bins=31, sub_policies=[])
    empty_policy_str = str(empty_policy)
    assert "sub_policies=[]" in empty_policy_str, empty_policy_str
    assert "augmentations={}" in empty_policy_str, empty_policy_str

    def get_first_augment():
        @augmentation(mag_range=(100, 200))
        def clashing_name(data, _):
            return data

        return clashing_name

    def get_second_augment():
        @augmentation
        def clashing_name(data, _):
            return data

        return clashing_name

    one = get_first_augment()
    another = get_second_augment()
    sub_policies = [
        [(one, 0.1, 5), (another, 0.4, None)],
        [(another, 0.2, None), (one, 0.5, 2)],
        [(another, 0.7, None)],
    ]
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
    def yet_another_aug(data, _):
        return data

    sub_policies = [
        [(yet_another_aug, 0.5, None), (one.augmentation(mag_range=(0, i)), 0.24, i)]
        for i in range(1, 107)
    ]
    bigger_policy = Policy(name="BiggerPolicy", num_magnitude_bins=200, sub_policies=sub_policies)
    for i, (first, second) in enumerate(bigger_policy.sub_policies):
        assert first[0].name == "000__yet_another_aug", f"{second[0].name}"
        assert second[0].name == f"{(i + 1):03}__clashing_name", f"{second[0].name}"


def test_unused_arg_fail():
    @pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0, seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy()
        return auto_augment.apply_auto_augment(image_net_policy, image, misspelled_kwarg=100)

    msg = "The kwarg `misspelled_kwarg` is not used by any of the augmentations."
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_empty_policy_fail():
    @pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0, seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        return auto_augment.apply_auto_augment(Policy("ShouldFail", 9, []), image)

    msg = (
        "Cannot run empty policy. Got Policy(name='ShouldFail', num_magnitude_bins=9, "
        "sub_policies=[], augmentations={}) in `apply_auto_augment` call."
    )
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_missing_shape_fail():
    @pipeline_def(enable_conditionals=True, batch_size=5, num_threads=4, device_id=0, seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        image_net_policy = auto_augment.get_image_net_policy(use_shape=True)
        return auto_augment.apply_auto_augment(image_net_policy, image)

    msg = "`translate_y` * provide it as `shape` argument to `apply_auto_augment` call"
    with assert_raises(Exception, glob=msg):
        pipeline()


def test_sub_policy_coalescing_matrix_correctness():
    @augmentation(
        mag_range=(0, 1),
        randomly_negate=True,
    )
    def first(data, op_id_mag_id):
        return op_id_mag_id

    @augmentation(
        mag_range=(1, 2),
        randomly_negate=True,
    )
    def second(data, op_id_mag_id):
        return op_id_mag_id

    @augmentation(
        mag_range=(2, 3),
    )
    def third(data, op_id_mag_id):
        return op_id_mag_id

    def custom_policy():
        return Policy(
            "MyCustomPolicy",
            14,
            [
                [(third, 0.5, 1), (second, 1.0, 2), (first, 0.2, 3), (second, 0.25, 4)],
                [(first, 0.6, 5)],
                [(second, 0.1, 6), (third, 0.7, 7)],
                [
                    (second, 0.8, 8),
                    (first, 0.9, 9),
                    (second, 1.0, 10),
                    (third, 0.2, 11),
                    (first, 0.5, 12),
                    (first, 1.0, 13),
                ],
            ],
        )

    predefined_policies = [
        auto_augment.get_image_net_policy,
        auto_augment.get_reduced_image_net_policy,
        auto_augment.get_svhn_policy,
        auto_augment.get_reduced_cifar10_policy,
        custom_policy,
    ]
    for policy_getter in predefined_policies:
        policy = policy_getter()
        matrix, augments = auto_augment._sub_policy_to_augmentation_matrix_map(policy)
        max_sub_policy_len = max(len(sub_policy) for sub_policy in policy.sub_policies)
        expected_shape = (len(policy.sub_policies), max_sub_policy_len)
        assert matrix.shape == expected_shape, f"{matrix.shape} {expected_shape} {policy}"
        for i, sub_policy in enumerate(policy.sub_policies):
            for stage_idx in range(max_sub_policy_len):
                mat_aug = augments[stage_idx][matrix[i, stage_idx]]
                if stage_idx < len(sub_policy):
                    sub_pol_aug, _, _ = sub_policy[stage_idx]
                    assert (
                        mat_aug is sub_pol_aug
                    ), f"{i} {stage_idx} {mat_aug} {sub_pol_aug} {policy}"
                else:
                    assert mat_aug is a.identity, f"{i} {stage_idx} {mat_aug} {policy}"


def test_wrong_sub_policy_format_fail():
    with assert_raises(
        Exception, glob="The `num_magnitude_bins` must be a positive integer, got 0"
    ):
        Policy("ShouldFail", 0.25, a.rotate)

    with assert_raises(
        Exception, glob="The `sub_policies` must be a list or tuple of sub policies"
    ):
        Policy("ShouldFail", 9, a.rotate)

    with assert_raises(Exception, glob="Each sub policy must be a list or tuple"):
        Policy("ShouldFail", 9, [a.rotate])

    with assert_raises(
        Exception, glob="as a triple: (augmentation, probability, magnitude). Got Augmentation"
    ):
        Policy("ShouldFail", 9, [(a.rotate, a.shear_x)])

    with assert_raises(Exception, glob="must be an instance of Augmentation. Got `0.5`"):
        Policy("ShouldFail", 9, [[(0.5, a.rotate, 3)]])

    with assert_raises(
        Exception, glob="Probability * must be a number from `[[]0, 1[]]` range. Got `2`"
    ):
        Policy("ShouldFail", 9, [[(a.rotate, 2, 2)]])

    with assert_raises(Exception, glob="Magnitude ** `[[]0, 8[]]` range. Got `-1`"):
        Policy("ShouldFail", 9, [[(a.rotate, 1, -1)]])

    @augmentation(mag_range=(0, 250))
    def parametrized_aug(data, magnitude):
        return data

    @augmentation
    def non_parametrized_aug(data, _):
        return data

    with assert_raises(Exception, glob="the magnitude bin is required"):
        Policy("ShouldFail", 7, [[(parametrized_aug, 0.5, None)]])

    with assert_warns(glob="probability 0 in one of the sub-policies"):
        Policy("ShouldFail", 7, [[(parametrized_aug, 0, 5)]])

    with assert_warns(glob="probability 0 in one of the sub-policies"):
        Policy("ShouldFail", 7, [[(parametrized_aug, 0.0, 5)]])

    with assert_warns(glob="The augmentation does not accept magnitudes"):
        Policy("ShouldFail", 7, [[(non_parametrized_aug, 1.0, 5)]])
