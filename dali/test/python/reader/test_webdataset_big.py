# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os
from glob import glob

import webdataset_base as base
from test_utils import compare_pipelines, get_dali_extra_path


def cross_check(
    dont_use_mmap,
    batch_size,
    num_shards,
    shard_id,
    skip_cached_images,
    pad_last_batch,
    stick_to_shard,
):
    num_multiplications = 4
    num_samples = 20 * num_multiplications
    tar_file_paths = [
        os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/cross.tar")
    ] * num_multiplications
    index_files = [base.generate_temp_index_file(tar_file_path) for tar_file_path in tar_file_paths]

    extract_dirs = [base.generate_temp_extract(tar_file_path) for tar_file_path in tar_file_paths]
    equivalent_files = sum(
        (
            sorted(
                glob(extract_dir.name + "/*"),
                key=lambda s: (int(s[s.rfind("/") + 1 : s.find(".")]), s),
            )
            for extract_dir in extract_dirs
        ),
        [],
    )

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_paths,
            [index_file.name for index_file in index_files],
            ["a.a;a.b;a.a;a.b", "b.a;b.b;b.a;b.b"],
            batch_size=batch_size,
            device_id=0,
            num_threads=10,
            dont_use_mmap=dont_use_mmap,
            num_shards=num_shards,
            shard_id=shard_id,
            prefetch_queue_depth=8,
            skip_cached_images=skip_cached_images,
            pad_last_batch=pad_last_batch,
            stick_to_shard=stick_to_shard,
        ),
        base.file_reader_pipeline(
            equivalent_files,
            ["a.a", "b.a"],
            batch_size=batch_size,
            device_id=0,
            num_threads=10,
            dont_use_mmap=True,
            num_shards=num_shards,
            shard_id=shard_id,
            skip_cached_images=skip_cached_images,
            pad_last_batch=pad_last_batch,
            stick_to_shard=stick_to_shard,
        ),
        batch_size,
        math.ceil(num_samples / base.test_batch_size),
    )


def test_cross_check():
    scenarios = [
        (
            dont_use_mmap,
            batch_size,
            num_shards,
            shard_id,
            skip_cached_images,
            pad_last_batch,
            stick_to_shard,
        )
        for dont_use_mmap in (False, True)
        for stick_to_shard in (False, True)
        for pad_last_batch in (False, True)
        for skip_cached_images in (False, True)
        for batch_size in (1, 8)
        if batch_size != 1 or not pad_last_batch
        for num_shards in (1, 80)
        for shard_id in {0, num_shards - 1}
    ]

    for args in scenarios:
        yield (cross_check,) + args
