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
from scipy.stats import chisquare
from nose2.tools import params

from nvidia.dali import fn, types
from nvidia.dali.pipeline import experimental
from nvidia.dali.auto_aug import trivial_augment
from nvidia.dali.auto_aug.core import augmentation
from test_utils import get_dali_extra_path

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


@params(*tuple(enumerate(itertools.product((True, False), (True, False), (None, 0),
                                           (True, False)))))
def test_run_trivial(i, args):
    uniformly_resized, use_shape, fill_value, specify_translation_bounds = args
    batch_sizes = [1, 8, 7, 64, 13, 64, 128]
    num_magnitude_bin_cases = [1, 11, 31, 40]
    batch_size = batch_sizes[i % len(batch_sizes)]
    num_magnitude_bins = num_magnitude_bin_cases[i % len(num_magnitude_bin_cases)]

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
        image = trivial_augment.trivial_augment_wide(image, num_magnitude_bins=num_magnitude_bins,
                                                     **extra)
        return image

    p = pipeline()
    p.build()
    for _ in range(3):
        p.run()


@params(*tuple(itertools.product((True, False), (0, 1), ('x', 'y'))))
def test_translation(use_shape, offset_fraction, extent):
    # make sure the translation helper processes the args properly
    # note, it only uses translate_y (as it is in imagenet policy)
    fill_value = 0
    params = {}
    if use_shape:
        param = offset_fraction
        param_name = "max_translate_rel"
    else:
        param = 1000 * offset_fraction
        param_name = "max_translate_abs"
    params[param_name] = param
    translation_x, translation_y = trivial_augment._get_translations(use_shape=use_shape, **params)
    augment = [translation_x] if extent == 'x' else [translation_y]

    @experimental.pipeline_def(enable_conditionals=True, batch_size=9, num_threads=4, device_id=0,
                               seed=43)
    def pipeline():
        encoded_image, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(encoded_image, device="mixed")
        if use_shape:
            shape = fn.peek_image_shape(encoded_image)
            return trivial_augment.apply_trivial_augment(augment, image, num_magnitude_bins=3,
                                                         fill_value=fill_value, shape=shape)
        else:
            return trivial_augment.apply_trivial_augment(augment, image, num_magnitude_bins=3,
                                                         fill_value=fill_value)

    p = pipeline()
    p.build()
    output, = p.run()
    output = [np.array(sample) for sample in output.as_cpu()]
    if offset_fraction == 1:
        # magnitudes are random here, but some should randomly be maximal
        all_black = 0
        for i, sample in enumerate(output):
            sample = np.array(sample)
            all_black += np.all(sample == fill_value)
        assert all_black
    else:
        for i, sample in enumerate(output):
            sample = np.array(sample)
            background_count = np.sum(sample == fill_value)
            assert background_count / sample.size < 0.1, \
                f"sample_idx: {i}, {background_count / sample.size}"


@params(*tuple(itertools.product(
    ['cpu', 'gpu'],
    [True, False],
    [1, 3, 7],
    [2, 3, 7],
)))
def test_ops_mags_selection(dev, use_sign, num_magnitude_bins, num_ops):
    # the chisquare expects at least 5 elements in a bin and we can have around
    # num_magnitude_bins * num_ops * (2**use_signs)
    batch_size = 2048

    def as_param_with_op_id(op_id):

        def as_param(magnitude):
            return np.array([op_id, magnitude], dtype=np.int32)

        return as_param

    @augmentation(param_device=dev)
    def op(sample, op_id_mag_id):
        return fn.cat(sample, op_id_mag_id)

    augmentations = [
        op.augmentation(mag_range=(10 * i + 1, 10 * i + num_magnitude_bins),
                        as_param=as_param_with_op_id(i + 1), randomly_negate=use_sign
                        and i % 3 == 0) for i in range(num_ops)
    ]

    expected_counts = {}
    prob = 1. / (num_ops * num_magnitude_bins)
    for aug in augmentations:
        magnitudes = aug._get_magnitudes(num_magnitude_bins)
        assert len(magnitudes) == num_magnitude_bins
        for mag in magnitudes:
            if not aug.randomly_negate:
                expected_counts[tuple(aug.as_param(mag))] = prob
            else:
                expected_counts[tuple(aug.as_param(mag))] = prob / 2
                expected_counts[tuple(aug.as_param(-mag))] = prob / 2
    expected_counts = {output: p * batch_size for output, p in expected_counts.items()}

    @experimental.pipeline_def(enable_conditionals=True, batch_size=batch_size, num_threads=4,
                               device_id=0, seed=42)
    def pipeline():
        sample = types.Constant([], dtype=types.INT32)
        if dev == "gpu":
            sample = sample.gpu()
        sample = trivial_augment.apply_trivial_augment(augmentations, sample,
                                                       num_magnitude_bins=num_magnitude_bins)
        return sample

    p = pipeline()
    p.build()
    stats = []
    for i in range(3):
        output, = p.run()
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
    assert 0.05 <= mean_p_val <= 0.95, f"{mean_p_val} {stat} {actual} {expected}"
