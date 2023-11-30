# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from test_utils import get_dali_extra_path

import numpy as np

data_root = get_dali_extra_path()
jpeg_file = os.path.join(data_root, "db", "single", "jpeg", "510", "ship-1083562_640.jpg")
batch_size = 4


def cb(sample_info):
    encoded_img = np.fromfile(jpeg_file, dtype=np.uint8)
    label = 1
    return encoded_img, np.int32([label])


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.external_source(source=cb, num_outputs=2, parallel=True, batch=False)
    images = fn.decoders.image(jpegs, device="cpu")
    return images, labels


def _test_no_segfault(method, workers_num):
    """
    This may cause segmentation fault on Python teardown if shared memory wrappers managed by the
    py_pool are garbage collected before pipeline's backend
    """
    pipe = simple_pipeline(
        py_start_method=method,
        py_num_workers=workers_num,
        batch_size=batch_size,
        num_threads=4,
        prefetch_queue_depth=2,
        device_id=0,
    )
    pipe.build()
    pipe.run()


def test_no_segfault():
    import multiprocessing
    import signal

    for method in ["fork", "spawn"]:
        # Repeat test a few times as garbage collection order failure is subject to race condition
        # and tended to exit properly once in a while
        for _ in range(2):
            for workers_num in range(1, 5):
                mp = multiprocessing.get_context("spawn")
                process = mp.Process(target=_test_no_segfault, args=(method, workers_num))
                process.start()
                process.join()
                if process.exitcode != os.EX_OK:
                    if signal.SIGSEGV == -process.exitcode:
                        raise RuntimeError("Process terminated with signal SIGSEGV")
                    raise RuntimeError("Process exited with {} code".format(process.exitcode))
