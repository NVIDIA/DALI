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

from nose2.tools import params

from nvidia.dali import fn
from nvidia.dali.pipeline import experimental
from nvidia.dali.auto_aug import trivial_augment
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
