#  Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# DALI pipeline definition
@pipeline_def
def naive_hist_pipe():
    img, _ = fn.readers.file(files=test_file_list)
    # The naive_histogram accepts single-channels image, thus we convert the image to Grayscale.
    img = fn.decoders.image(img, device="mixed", output_type=DALIImageType.GRAY)
    img = img.gpu()
    img = fn.naive_histogram(img, n_bins=24)
    return img


def test_naive_histogram():
    pipe = naive_hist_pipe(batch_size=2, num_threads=1, device_id=0)
    out = pipe.run()
    print(out[0].as_cpu().as_array())
