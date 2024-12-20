# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali as dali
import os
import tempfile
from glob import glob
from nose_utils import assert_equals, assert_raises

import webdataset_base as base
from test_utils import compare_pipelines, get_dali_extra_path


def general_corner_case(
    test_batch_size=base.test_batch_size, dtypes=None, missing_component_behavior="", **kwargs
):
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "cls"],
            missing_component_behavior=missing_component_behavior,
            dtypes=dtypes,
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs,
        ),
        base.file_reader_pipeline(
            equivalent_files,
            ["jpg", "cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_mmap_dtype_incompatibility():
    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=[dali.types.INT8, dali.types.FLOAT64],
        glob="component size and dtype incompatible",
    )


def test_lazy_init():
    general_corner_case(lazy_init=True)


def test_read_ahead():
    general_corner_case(read_ahead=True)


def test_single_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            missing_component_behavior="skip",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_single_sample_and_junk():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single_junk.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_wide_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/wide.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    num_components = 1000
    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_argument_errors():
    def paths_index_paths_error():
        webdataset_pipeline = base.webdataset_raw_pipeline(
            [
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
            ],
            ["test.idx"],
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        paths_index_paths_error,
        glob="The number of index files, if any, must match the number of archives in the dataset",
    )

    assert_raises(
        RuntimeError,
        general_corner_case,
        missing_component_behavior="SomethingInvalid",
        glob="Invalid value for missing_component_behavior",
    )
    general_corner_case(missing_component_behavior="Skip")

    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=[dali.types.STRING, dali.types.STRING],
        glob="Unsupported output dtype *. Supported types are",
    )
    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=dali.types.INT8,
        glob="Number of extensions does not match the number of provided types",
    )


def general_index_error(
    index_file_contents, tar_file_path="db/webdataset/MNIST/devel-0.tar", ext="jpg"
):
    index_file = tempfile.NamedTemporaryFile()
    index_file.write(index_file_contents)
    index_file.flush()
    webdataset_pipeline = base.webdataset_raw_pipeline(
        os.path.join(get_dali_extra_path(), tar_file_path),
        index_file.name,
        ext,
        batch_size=1,
        device_id=0,
        num_threads=1,
    )
    webdataset_pipeline.run()
    webdataset_pipeline.run()


def test_index_errors():
    assert_raises(RuntimeError, general_index_error, b"", glob="no version signature found")
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v0.1",
        glob="Unsupported version of the index file (v0.1).",
    )
    assert_raises(RuntimeError, general_index_error, b"v1.1", glob="no sample count found")
    assert_raises(
        RuntimeError, general_index_error, b"v1.1 -1", glob="sample count must be positive"
    )
    assert_raises(
        RuntimeError, general_index_error, b"v1.1 1\n", glob="no extensions provided for the sample"
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg",
        glob="Could not find all necessary component parameters",
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg 1 0",
        glob="tar offset is not a multiple of tar block size",
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg 51200 1",
        "db/webdataset/sample-tar/empty.tar",
        glob="offset is outside of the archive file",
    )
