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
from nvidia.dali.auto_aug import trivial_augment
from nvidia.dali.auto_aug.core import augmentation
from test_utils import get_dali_extra_path, check_batch

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")
vid_dir = os.path.join(data_root, "db", "video", "sintel", "video_files")
vid_files = ["sintel_trailer-720p_2.mp4"]
vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]


@params(
    *tuple(
        enumerate(
            itertools.product(
                ("cpu", "gpu"), (True, False), (True, False), (None, 0), (True, False)
            )
        )
    )
)
def test_run_trivial(i, args):
    dev, uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 64, 41]
    num_magnitude_bin_cases = [1, 11, 31, 40]
    batch_size = batch_sizes[i % len(batch_sizes)]
    num_magnitude_bins = num_magnitude_bin_cases[i % len(num_magnitude_bin_cases)]

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
        image = trivial_augment.trivial_augment_wide(
            image, num_magnitude_bins=num_magnitude_bins, **extra
        )
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
        size_1 = (223, 367)
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
                    ("cpu", 6, False, 1),
                    ("cpu", 1, True, 10),
                    ("gpu", 12, True, None),
                    ("gpu", 4, False, 101),
                )
            )
        )
    )
    def test_uniform(self, i, args):
        device, batch_size, use_shape, num_magnitude_bins = args
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
            if num_magnitude_bins is not None:
                extra["num_magnitude_bins"] = num_magnitude_bins
            if device == "gpu":
                video = video.gpu()
            video = trivial_augment.trivial_augment_wide(video, **extra)
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
        itertools.product(
            ["cpu", "gpu"],
            [True, False],
            [1, 3, 7],
            [2, 3, 7],
        )
    )
)
def test_ops_mags_selection(dev, use_sign, num_magnitude_bins, num_ops):
    # the chisquare expects at least 5 elements in a bin and we can have around
    # num_magnitude_bins * num_ops * (2**use_signs)
    batch_size = 2048

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
    prob = 1.0 / (num_ops * num_magnitude_bins)
    for aug in augmentations:
        magnitudes = aug._get_magnitudes(num_magnitude_bins)
        assert len(magnitudes) == num_magnitude_bins
        for mag in magnitudes:
            if not aug.randomly_negate:
                expected_counts[tuple(aug.mag_to_param(mag))] = prob
            else:
                expected_counts[tuple(aug.mag_to_param(mag))] = prob / 2
                expected_counts[tuple(aug.mag_to_param(-mag))] = prob / 2
    expected_counts = {output: p * batch_size for output, p in expected_counts.items()}

    @pipeline_def(
        enable_conditionals=True, batch_size=batch_size, num_threads=4, device_id=0, seed=42
    )
    def pipeline():
        data = types.Constant([], dtype=types.INT32)
        if dev == "gpu":
            data = data.gpu()
        data = trivial_augment.apply_trivial_augment(
            augmentations, data, num_magnitude_bins=num_magnitude_bins
        )
        return data

    p = pipeline()
    stats = []
    for i in range(3):
        (output,) = p.run()
        output = [np.array(s) for s in (output.as_cpu() if dev == "gpu" else output)]
        actual_count = {allowed_out: 0 for allowed_out in expected_counts}
        for sample in output:
            actual_count[tuple(sample)] += 1
        actual = []
        expected = []
        for out in expected_counts:
            actual.append(actual_count[out])
            expected.append(expected_counts[out])
        stat = chisquare(actual, expected)
        stats.append(stat)
    mean_p_val = sum(stat.pvalue for stat in stats) / len(stats)
    assert 0.01 <= mean_p_val, f"{mean_p_val} {stat} {actual} {expected}"
