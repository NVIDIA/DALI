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
import random

import unittest
import numpy as np
from scipy.stats import chisquare
from nose2.tools import params

from nvidia.dali import fn, types
from nvidia.dali import pipeline_def
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.auto_aug.core import augmentation

from test_utils import get_dali_extra_path, check_batch
from nose_utils import assert_raises

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")
vid_dir = os.path.join(data_root, "db", "video", "sintel", "video_files")
vid_files = ["sintel_trailer-720p_3.mp4"]
vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]


def debug_discrepancy_helper(*batch_pairs):
    """
    Accepts list of triples: left_batch, right_batch, name of a batch.
    Prepares a list of statistics for any differences between samples in the corresponding batches.
    """

    def as_array_list(batch):
        batch = batch.as_cpu()
        return [np.array(sample) for sample in batch]

    batch_names = [name for _, _, name in batch_pairs]
    batch_pairs = [(as_array_list(left), as_array_list(right)) for left, right, _ in batch_pairs]
    batch_stats = []
    for batch_name, batch_pair in zip(batch_names, batch_pairs):
        left, right = batch_pair
        num_samples = len(left), len(right)
        sample_diffs = []
        for sample_idx, (sample_left, sample_right) in enumerate(zip(left, right)):
            if sample_left.shape != sample_right.shape:
                sample_diffs.append(
                    {
                        "sample_idx": sample_idx,
                        "sample_left_shape": sample_left.shape,
                        "sample_right_shape": sample_right.shape,
                    }
                )
            else:
                absdiff = np.maximum(sample_right, sample_left) - np.minimum(
                    sample_right, sample_left
                )
                err = np.mean(absdiff)
                max_err = np.max(absdiff)
                min_err = np.min(absdiff)
                total_errors = np.sum(absdiff != 0)
                if any(val != 0 for val in (err, max_err, max_err, total_errors)):
                    sample_diffs.append(
                        {
                            "sample_idx": sample_idx,
                            "mean_error": err,
                            "max_error": max_err,
                            "min_err": min_err,
                            "total_errors": total_errors,
                            "shape": sample_left.shape,
                        }
                    )
        batch_stats.append(
            {"batch_name": batch_name, "num_samples": num_samples, "sample_diffs": sample_diffs}
        )
    return batch_stats


@params(
    *tuple(
        enumerate(
            itertools.product(
                ("cpu", "gpu"), (True, False), (True, False), (None, 0), (True, False)
            )
        )
    )
)
def test_run_rand_aug(i, args):
    dev, uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    # Keep batch_sizes ns and ms length co-prime
    batch_sizes = [1, 8, 7, 13, 31, 64, 47]
    ns = [1, 2, 3]
    ms = [0, 12, 15, 30]
    batch_size = batch_sizes[i % len(batch_sizes)]
    n = ns[i % len(ns)]
    m = ms[i % len(ms)]

    @pipeline_def(
        enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0, seed=43
    )
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        decoded_image = fn.decoders.image(encoded_image, device="cpu" if dev == "cpu" else "mixed")
        resized_image = (
            decoded_image if not uniformly_resized else fn.resize(decoded_image, size=(244, 244))
        )
        extra = {} if not use_shape else {"shape": fn.peek_image_shape(encoded_image)}
        if fill_value is not None:
            extra["fill_value"] = fill_value
        if specify_translation_bounds:
            if use_shape:
                extra["max_translate_rel"] = 0.9
            else:
                extra["max_translate_abs"] = 400
        raugmented_image = rand_augment.rand_augment(resized_image, n=n, m=m, **extra)
        return encoded_image, decoded_image, resized_image, raugmented_image

    # run the pipeline twice to make sure instantiation preserves determinism
    p1 = pipeline()
    p2 = pipeline()
    for iteration_idx in range(3):
        encoded1, decoded1, resized1, out1 = p1.run()
        encoded2, decoded2, resized2, out2 = p2.run()
        try:
            check_batch(out1, out2)
        except AssertionError as e:
            diffs = debug_discrepancy_helper(
                (encoded1, encoded2, "encoded"),
                (decoded1, decoded2, "decoded"),
                (resized1, resized2, "resized"),
                (out1, out2, "out"),
            )
            iter_diff = {"iteration_idx": iteration_idx, "diffs": diffs}
            raise AssertionError(
                f"The outputs do not match, the differences between encoded, decoded, "
                f"resized and augmented batches are respectively: {repr(iter_diff)}"
            ) from e


class VideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        num_frames = 31
        roi_start = (90, 0)
        roi_end = (630, 1280)
        size_1 = (223, 367)
        size_2 = (400, 100)

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
                    ("cpu", 4, False, 2, 8, True),
                    ("cpu", 2, True, 2, 10, False),
                    ("gpu", 7, False, 3, 5, True),
                    ("gpu", 1, True, 1, 7, True),
                )
            )
        )
    )
    def test_uniform(self, i, args):
        device, batch_size, use_shape, n, m, monotonic_mag = args
        num_iterations = 3

        assert device in ("gpu", "cpu")

        @pipeline_def(
            batch_size=batch_size, device_id=0, num_threads=4, seed=42, enable_conditionals=True
        )
        def pipeline():
            rng = random.Random(42 + i)
            video = fn.external_source(
                source=lambda: list(rng.choices(self.vid_files, k=batch_size)),
                batch=True,
                layout="FHWC",
            )
            extra = {} if not use_shape else {"shape": video.shape()[1:]}
            extra["monotonic_mag"] = monotonic_mag
            if device == "gpu":
                video = video.gpu()
            video = rand_augment.rand_augment(video, n=n, m=m, **extra)
            return video

        # run the pipeline twice to make sure instantiation preserves determinism
        p1 = pipeline()
        p2 = pipeline()

        for _ in range(num_iterations):
            (out1,) = p1.run()
            (out2,) = p2.run()
            check_batch(out1, out2)


@params(
    *tuple(
        enumerate(
            itertools.product(
                ["cpu", "gpu"],
                [True, False],
                [1, 2, 3],
                [2, 3],
            )
        )
    )
)
def test_ops_selection_and_mags(case_idx, args):
    dev, use_sign, n, num_ops = args
    num_magnitude_bins = 9
    # the chisquare expects at least 5 elements in a bin and we can have around
    # (num_ops * (2**use_signs)) ** n ops
    batch_size = 2048
    magnitude_cases = list(range(num_magnitude_bins))
    m = magnitude_cases[case_idx % len(magnitude_cases)]

    def mag_to_param_with_op_id(op_id):
        def mag_to_param(magnitude):
            return np.array([op_id, magnitude], dtype=np.int32)

        return mag_to_param

    @augmentation(param_device=dev)
    def op(data, op_id_mag_id):
        return fn.cat(data, op_id_mag_id)

    augmentations = [
        op.augmentation(
            mag_range=(10 * i + 1, 10 * i + num_magnitude_bins),
            mag_to_param=mag_to_param_with_op_id(i + 1),
            randomly_negate=use_sign and i % 3 == 0,
        )
        for i in range(num_ops)
    ]

    expected_counts = {}
    seq_prob = 1.0 / (num_ops**n)
    for aug_sequence in itertools.product(*([augmentations] * n)):
        possible_signs = [(-1, 1) if aug.randomly_negate else (1,) for aug in aug_sequence]
        possible_signs = tuple(itertools.product(*possible_signs))
        prob = seq_prob / len(possible_signs)
        for signs in possible_signs:
            assert len(aug_sequence) == len(signs)
            outs = []
            for aug, sign in zip(aug_sequence, signs):
                mag = aug._get_magnitudes(num_magnitude_bins)[m]
                op_id_mag = aug.mag_to_param(mag * sign)
                outs.append(op_id_mag)
            expected_counts[tuple(el for out in outs for el in out)] = prob
    expected_counts = {output: p * batch_size for output, p in expected_counts.items()}

    @pipeline_def(
        enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0, seed=42
    )
    def pipeline():
        data = types.Constant([], dtype=types.INT32)
        if dev == "gpu":
            data = data.gpu()
        data = rand_augment.apply_rand_augment(
            augmentations, data, n=n, m=m, num_magnitude_bins=num_magnitude_bins
        )
        return fn.reshape(data, shape=(-1, 2))

    p = pipeline()
    for i in range(3):
        (output,) = p.run()
        output = [np.array(s) for s in (output.as_cpu() if dev == "gpu" else output)]
        actual_count = {allowed_out: 0 for allowed_out in expected_counts}
        for sample in output:
            assert len(sample) == n, f"{i} {sample}"
            out = tuple(el for op_mag in sample for el in op_mag)
            actual_count[out] += 1
        actual = []
        expected = []
        for out in expected_counts:
            actual.append(actual_count[out])
            expected.append(expected_counts[out])
        stat = chisquare(actual, expected)
        assert 0.01 <= stat.pvalue, f"{stat} {actual} {expected}"


def test_wrong_params_fail():
    @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
    def pipeline(n, m, num_magnitude_bins):
        data = types.Constant(np.array([[[]]], dtype=np.uint8))
        return rand_augment.rand_augment(data, n=n, m=m, num_magnitude_bins=num_magnitude_bins)

    with assert_raises(
        Exception, glob="The number of operations to apply `n` must be a non-negative integer"
    ):
        pipeline(n=None, m=1, num_magnitude_bins=11)

    with assert_raises(Exception, glob="The `num_magnitude_bins` must be a positive integer, got"):
        pipeline(n=1, m=1, num_magnitude_bins=None)

    with assert_raises(Exception, glob="`m` must be an integer from `[[]0, 14[]]` range. Got 15."):
        pipeline(n=1, m=15, num_magnitude_bins=15)

    with assert_raises(Exception, glob="The `augmentations` list cannot be empty"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def no_aug_pipeline():
            data = types.Constant(np.array([[[]]], dtype=np.uint8))
            return rand_augment.apply_rand_augment([], data, 1, 20)

        no_aug_pipeline()

    with assert_raises(Exception, glob="The augmentation `translate_x` requires `shape` argument"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def missing_shape():
            data = types.Constant(np.array([[[]]], dtype=np.uint8))
            augments = rand_augment.get_rand_augment_suite(use_shape=True)
            return rand_augment.apply_rand_augment(augments, data, 1, 20)

        missing_shape()

    with assert_raises(Exception, glob="The kwarg `shhape` is not used by any of the"):

        @pipeline_def(batch_size=4, device_id=0, num_threads=4, seed=42, enable_conditionals=True)
        def unused_kwarg():
            data = types.Constant(np.array([[[]]], dtype=np.uint8))
            augments = rand_augment.get_rand_augment_suite(use_shape=True)
            return rand_augment.apply_rand_augment(augments, data, 1, 20, shhape=42)

        unused_kwarg()
