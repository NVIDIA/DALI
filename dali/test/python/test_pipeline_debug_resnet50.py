# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
from itertools import product
from nvidia.dali.pipeline.experimental import pipeline_def
from time import time

from test_utils import get_dali_extra_path


@pipeline_def(device_id=0)
def rn50_pipeline(data_path):
    uniform = fn.random.uniform(range=(0.0, 1.0), shape=2)
    resize_uniform = fn.random.uniform(range=(256.0, 480.0))
    mirror = fn.random.coin_flip(probability=0.5)
    jpegs, _ = fn.readers.file(file_root=data_path)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    resized_images = fn.fast_resize_crop_mirror(
        images,
        crop=(224, 224),
        crop_pos_x=uniform[0],
        crop_pos_y=uniform[1],
        mirror=mirror,
        resize_shorter=resize_uniform,
    )
    output = fn.crop_mirror_normalize(
        resized_images.gpu(),
        device="gpu",
        dtype=types.FLOAT16,
        mean=[128.0, 128.0, 128.0],
        std=[1.0, 1.0, 1.0],
    )
    return output


@pipeline_def(device_id=0)
def rn50_pipeline_2(data_path):
    uniform = fn.random.uniform(range=(0.0, 1.0), shape=2)
    resize_uniform = fn.random.uniform(range=(256.0, 480.0))
    mirror = fn.random.coin_flip(probability=0.5)
    jpegs, _ = fn.readers.file(file_root=data_path)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    resized_images = fn.resize(
        images, device="gpu", interp_type=types.INTERP_LINEAR, resize_shorter=resize_uniform
    )
    output = fn.crop_mirror_normalize(
        resized_images,
        device="gpu",
        dtype=types.FLOAT16,
        crop=(224, 224),
        mean=[128.0, 128.0, 128.0],
        std=[1.0, 1.0, 1.0],
        mirror=mirror,
        crop_pos_x=uniform[0],
        crop_pos_y=uniform[1],
    )
    return output


def run_benchmark(pipe_fun, batch_size, num_threads, num_samples, debug, data_path):
    num_iters = num_samples // batch_size
    times = np.empty(num_iters + 1)

    times[0] = time()
    pipe = pipe_fun(data_path, batch_size=batch_size, num_threads=num_threads, debug=debug)
    build_time = time()

    for i in range(num_iters):
        pipe.run()
        times[i + 1] = time()

    full_time = times[-1] - build_time
    times = np.diff(times)

    return full_time, times[0], times[1:]


def test_rn50_benchmark(
    pipe_fun=rn50_pipeline,
    batch_size=8,
    num_threads=2,
    num_samples=256,
    data_path=None,
    save_df=None,
):
    if not data_path:
        data_path = os.path.join(get_dali_extra_path(), "db/single/jpeg")

    print(f"num_threads: {num_threads}, batch_size: {batch_size}")

    full_stand, build_stand, times_stand = run_benchmark(
        pipe_fun, batch_size, num_threads, num_samples, False, data_path
    )
    iter_time_stand = np.mean(times_stand[1:]) / batch_size
    avg_speed_stand = num_samples / full_stand

    print(
        f"Stand pipeline --- time: {full_stand:8.5f} [s] --- "
        f"build + 1st iter time: {build_stand:.5f} [s] --- "
        f"avg iter time per sample: {iter_time_stand:7.5f} [s] --- "
        f"avg speed: {avg_speed_stand:8.3f} [img/s]"
    )

    full_debug, build_debug, times_debug = run_benchmark(
        pipe_fun, batch_size, num_threads, num_samples, True, data_path
    )
    iter_time_debug = np.mean(times_debug[1:]) / batch_size
    avg_speed_debug = num_samples / full_debug

    print(
        f"Debug pipeline --- time: {full_debug:8.5f} [s] --- "
        f"build + 1st iter time: {build_debug:.5f} [s] --- "
        f"avg iter time per sample: {iter_time_debug:7.5f} [s] --- "
        f"avg speed: {avg_speed_debug:8.3f} [img/s]"
    )

    if save_df is not None:
        df = pd.DataFrame(
            {
                "type": ["standard_sync", "debug_old"],
                "batch_size": batch_size,
                "time": [full_stand, full_debug],
                "iter_time": [iter_time_stand, iter_time_debug],
                "avg_speed": [avg_speed_stand, avg_speed_debug],
            }
        )
        return pd.concat([save_df, df])

    return None


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 32, 64, 128],
        help="List of batch sizes to run",
    )
    parser.add_argument(
        "--thread-counts", nargs="+", type=int, default=[1, 2, 4, 8], help="List of thread counts"
    )
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of samples")
    parser.add_argument("--data-path", type=str, help="Directory path of training dataset")
    parser.add_argument("--save-dir", type=str, help="Directory where to save results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = None
    for pipe_fun, num_threads in product([rn50_pipeline, rn50_pipeline_2], args.thread_counts):
        if args.save_dir is not None:
            import pandas as pd

            save_file = os.path.join(
                args.save_dir, f"bench_{pipe_fun.__name__}_threads_{num_threads}.csv"
            )
            if os.path.isfile(save_file):
                df = pd.read_csv(save_file)
            else:
                df = pd.DataFrame(columns=["type", "batch_size", "time", "iter_time", "avg_speed"])

        for batch_size in args.batch_sizes:
            df = test_rn50_benchmark(
                rn50_pipeline_2, batch_size, num_threads, args.num_samples, args.data_path, df
            )

        if df is not None:
            df.to_csv(save_file, index=False)
