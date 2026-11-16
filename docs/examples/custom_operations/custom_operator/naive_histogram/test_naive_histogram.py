#  Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import numpy as np
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIImageType
import nvidia.dali.fn as fn

# Load the Custom Operator
import nvidia.dali.plugin_manager as plugin_manager

plugin_manager.load_library("./build/libnaivehistogram.so")


# List test files. This step should be customized.
dali_extra_path = os.environ["DALI_EXTRA_PATH"]
test_file_list = [
    dali_extra_path + "/db/single/jpeg/100/swan-3584559_640.jpg",
    dali_extra_path + "/db/single/jpeg/113/snail-4368154_1280.jpg",
    dali_extra_path + "/db/single/jpeg/100/swan-3584559_640.jpg",
    dali_extra_path + "/db/single/jpeg/113/snail-4368154_1280.jpg",
    dali_extra_path + "/db/single/jpeg/100/swan-3584559_640.jpg",
    dali_extra_path + "/db/single/jpeg/113/snail-4368154_1280.jpg",
]


N_BINS = 24


# DALI pipeline definition
@pipeline_def
def naive_hist_pipe():
    jpegs, _ = fn.readers.file(files=test_file_list)
    # The naive_histogram accepts single-channels image, thus we convert the image to Grayscale.
    img = fn.decoders.image(
        jpegs, device="mixed", output_type=DALIImageType.GRAY
    )
    hist = fn.naive_histogram(img, n_bins=N_BINS)
    return img, hist


def test_naive_histogram():
    batch_size = 2
    pipe = naive_hist_pipe(batch_size=batch_size, num_threads=1, device_id=0)
    # Run several iterations, so that the output memory gets reused.
    for _ in range(len(test_file_list) // batch_size):
        imgs, hists = pipe.run()
        imgs, hists = imgs.as_cpu(), hists.as_cpu()
        for i in range(batch_size):
            img, hist = imgs.at(i), hists.at(i)
            # The kernel puts a pixel of value `v` into the bin `v % n_bins`.
            ref = np.bincount(img.ravel() % N_BINS, minlength=N_BINS)
            assert np.array_equal(
                hist, ref
            ), f"Got histogram {hist}, expected {ref}."
    print(hists.as_array())


if __name__ == "__main__":
    test_naive_histogram()
