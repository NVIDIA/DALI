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

from nvidia.dali import pipeline_def
from nvidia.dali.fn import readers
from nose_utils import assert_equals
import tempfile
from subprocess import call
import os
import tarfile

test_batch_size = 4
wds2idx_script = "../../../tools/wds2idx.py"


@pipeline_def()
def webdataset_raw_pipeline(
    paths,
    index_paths,
    ext,
    case_sensitive_extensions=True,
    missing_component_behavior="empty",
    dtypes=None,
    dont_use_mmap=False,
    num_shards=1,
    shard_id=0,
    skip_cached_images=False,
    pad_last_batch=False,
    lazy_init=False,
    read_ahead=False,
    stick_to_shard=False,
):
    out = readers.webdataset(
        paths=paths,
        index_paths=index_paths,
        ext=ext,
        case_sensitive_extensions=case_sensitive_extensions,
        missing_component_behavior=missing_component_behavior,
        dtypes=dtypes,
        dont_use_mmap=dont_use_mmap,
        prefetch_queue_depth=1,
        num_shards=num_shards,
        shard_id=shard_id,
        stick_to_shard=stick_to_shard,
        skip_cached_images=skip_cached_images,
        pad_last_batch=pad_last_batch,
        lazy_init=lazy_init,
        read_ahead=read_ahead,
    )
    return out if not isinstance(out, list) else tuple(out)


def filter_ext(files, exts):
    if isinstance(exts, str):
        exts = {exts}
    return list(filter(lambda s: any(map(lambda ext: s.endswith("." + ext), exts)), files))


@pipeline_def()
def file_reader_pipeline(
    files,
    exts=None,
    dont_use_mmap=False,
    num_shards=1,
    shard_id=0,
    skip_cached_images=False,
    pad_last_batch=False,
    lazy_init=False,
    read_ahead=False,
    stick_to_shard=False,
):
    if not isinstance(exts, list):
        exts = [exts]

    return tuple(
        (
            readers.file(
                files=filter_ext(files, ext),
                dont_use_mmap=dont_use_mmap,
                prefetch_queue_depth=1,
                num_shards=num_shards,
                shard_id=shard_id,
                stick_to_shard=stick_to_shard,
                skip_cached_images=skip_cached_images,
                pad_last_batch=pad_last_batch,
                lazy_init=lazy_init,
                read_ahead=read_ahead,
            )[0]
            if type(ext) in {str, set}
            else ext
        )
        for ext in exts
    )


def generate_temp_index_file(tar_file_path):
    temp_index_file = tempfile.NamedTemporaryFile()
    assert_equals(
        call([wds2idx_script, tar_file_path, temp_index_file.name], stdout=open(os.devnull, "wb")),
        0,
    )
    return temp_index_file


def generate_temp_extract(tar_file_path):
    temp_extract_dir = tempfile.TemporaryDirectory()
    archive = tarfile.open(tar_file_path)
    for member in archive:
        if member.type != tarfile.REGTYPE:
            continue
        archive.extract(member, temp_extract_dir.name)
    return temp_extract_dir
