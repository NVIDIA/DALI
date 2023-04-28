# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import timeit
from astropy.io import fits
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali

# @pipeline_def
def FitsReaderPipeline(device="cpu", batch_size=1, num_threads=1, file_list=None):
    data = fn.experimental.readers.fits(device=device, file_list=file_list, shard_id=0, num_shards=1)
    return data

def get_fits_filenames():
    dir_path = os.path.join(os.path.dirname(__file__), "test_fits_data")
    filenames = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".fits")]
    return filenames

def test_fits_speed():
    filenames = get_fits_filenames()
    if not filenames:
        print("No FITS files in this directory\n")
        return

    file_list_arg = os.path.join(os.path.dirname(__file__), "test_fits_data/file_list.txt")

    with open(file_list_arg, 'w') as f:
        for file in filenames:
                f.write(f"{file}\n")

    device = "cpu"
    batch_size=10
    num_threads=3

    pipe = nvidia.dali.Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    with pipe:
        data = FitsReaderPipeline(device=device, batch_size=batch_size, num_threads=num_threads, file_list=file_list_arg)
        pipe.set_outputs(data)

    times = []
    try:
        pipe.build()
        for i in range(len(filenames)):
            time_taken = timeit.timeit(pipe.run, number=1)
            times.append(time_taken)
            print("Time taken for file {}: {:.6f} s".format(i+1, time_taken))
    finally:
        del pipe

    os.remove(file_list_arg)

    total_time = sum(times)
    mean_time = total_time / len(filenames)
    variance = sum((time - mean_time) ** 2 for time in times) / len(times)
    print(device + ": Total time for all files: {:.6f} s".format(total_time))
    print(device + ": Mean time for a single file: {:.6f} s".format(mean_time))
    print(device + ": Variance: {:.6f}".format(variance))


if __name__ == '__main__':
    test_fits_speed()