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

import os
from glob import glob
import math
import nvidia.dali as dali
from test_utils import compare_pipelines, get_dali_extra_path
from nose_utils import assert_raises, assert_equals
from webdataset_base import (
    generate_temp_extract,
    generate_temp_index_file,
    webdataset_raw_pipeline,
    file_reader_pipeline,
)

from webdataset_base import test_batch_size  # noqa:F401, this is a parameter used in tests


def test_return_empty():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = glob(extract_dir.name + "/*")
    equivalent_files = sorted(
        equivalent_files, key=(lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")]))
    )  # noqa: 203

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "txt"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            missing_component_behavior="empty",
        ),
        file_reader_pipeline(
            equivalent_files, ["jpg", []], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_skip_sample():
    num_samples = 500
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(
        filter(
            lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")]) < 2500,  # noqa: 203
            sorted(
                glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
            ),  # noqa: 203
        )
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "cls"],
            missing_component_behavior="skip",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, ["jpg", "cls"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )
    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["jpg", "cls"],
        missing_component_behavior="skip",
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_raise_error_on_missing():
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar")
    index_file = generate_temp_index_file(tar_file_path)
    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["jpg", "cls"],
        missing_component_behavior="error",
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_raises(RuntimeError, wds_pipeline.build, glob="Underful sample detected")


def test_different_components():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/scrambled.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = glob(extract_dir.name + "/*")
    equivalent_files = sorted(
        equivalent_files, key=(lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")]))
    )  # noqa: 203

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "txt;cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files,
            ["jpg", {"txt", "cls"}],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_dtypes():
    num_samples = 100
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/dtypes.tar")
    index_file = generate_temp_index_file(tar_file_path)

    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["float16", "int32", "float64"],
        dtypes=[dali.types.FLOAT16, dali.types.INT32, dali.types.FLOAT64],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    for sample_idx in range(num_samples):
        if sample_idx % test_batch_size == 0:
            f16, i32, f64 = wds_pipeline.run()
        assert (f16.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()
        assert (i32.as_array()[sample_idx % test_batch_size] == [int(sample_idx)] * 10).all()
        assert (f64.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()


def test_wds_sharding():
    num_samples = 3000
    tar_file_paths = [
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
    ]
    index_files = [generate_temp_index_file(tar_file_path) for tar_file_path in tar_file_paths]

    extract_dirs = [generate_temp_extract(tar_file_path) for tar_file_path in tar_file_paths]
    equivalent_files = sum(
        list(
            sorted(
                glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
            )  # noqa: 203
            for extract_dir in extract_dirs
        ),
        [],
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_paths,
            [index_file.name for index_file in index_files],
            ["jpg", "cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files,
            ["jpg", "cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_sharding():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )  # noqa: 203

    num_shards = 100
    for shard_id in range(num_shards):
        compare_pipelines(
            webdataset_raw_pipeline(
                tar_file_path,
                index_file.name,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            file_reader_pipeline(
                equivalent_files,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            test_batch_size,
            math.ceil(num_samples / num_shards / test_batch_size) * 2,
        )


def test_pax_format():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    pax_tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/pax/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    num_shards = 100
    for shard_id in range(num_shards):
        compare_pipelines(
            webdataset_raw_pipeline(
                tar_file_path,
                index_file.name,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            webdataset_raw_pipeline(
                pax_tar_file_path,
                None,
                ext=["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            test_batch_size,
            math.ceil(num_samples / num_shards / test_batch_size) * 2,
        )


def test_case_sensitive_container_format():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    case_insensitive_tar_file_path = os.path.join(
        get_dali_extra_path(), "db/webdataset/case_insensitive/devel-0.tar"
    )
    index_file = generate_temp_index_file(tar_file_path)

    num_shards = 100
    with assert_raises(RuntimeError, glob="Underful sample detected at"):
        for shard_id in range(num_shards):
            compare_pipelines(
                webdataset_raw_pipeline(
                    tar_file_path,
                    index_file.name,
                    ["jpg", "cls"],
                    num_shards=num_shards,
                    shard_id=shard_id,
                    batch_size=test_batch_size,
                    device_id=0,
                    num_threads=1,
                ),
                webdataset_raw_pipeline(
                    case_insensitive_tar_file_path,
                    None,
                    ext=["jpg", "cls"],
                    missing_component_behavior="error",
                    num_shards=num_shards,
                    shard_id=shard_id,
                    batch_size=test_batch_size,
                    device_id=0,
                    num_threads=1,
                ),
                test_batch_size,
                math.ceil(num_samples / num_shards / test_batch_size) * 2,
            )


def test_case_sensitive_arg_format():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    num_shards = 100
    with assert_raises(RuntimeError, glob="Underful sample detected at"):
        for shard_id in range(num_shards):
            compare_pipelines(
                webdataset_raw_pipeline(
                    tar_file_path,
                    index_file.name,
                    ["jpg", "cls"],
                    num_shards=num_shards,
                    shard_id=shard_id,
                    batch_size=test_batch_size,
                    device_id=0,
                    num_threads=1,
                ),
                webdataset_raw_pipeline(
                    tar_file_path,
                    index_file.name,
                    ext=["Jpg", "cls"],
                    missing_component_behavior="error",
                    num_shards=num_shards,
                    shard_id=shard_id,
                    batch_size=test_batch_size,
                    device_id=0,
                    num_threads=1,
                ),
                test_batch_size,
                math.ceil(num_samples / num_shards / test_batch_size) * 2,
            )


def test_case_insensitive_container_format():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    case_insensitive_tar_file_path = os.path.join(
        get_dali_extra_path(), "db/webdataset/case_insensitive/devel-0.tar"
    )
    index_file = generate_temp_index_file(tar_file_path)

    num_shards = 100
    for shard_id in range(num_shards):
        compare_pipelines(
            webdataset_raw_pipeline(
                tar_file_path,
                index_file.name,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            webdataset_raw_pipeline(
                case_insensitive_tar_file_path,
                None,
                ext=["jpg", "cls"],
                case_sensitive_extensions=False,
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            test_batch_size,
            math.ceil(num_samples / num_shards / test_batch_size) * 2,
        )


def test_case_insensitive_arg_format():
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    num_shards = 100
    for shard_id in range(num_shards):
        compare_pipelines(
            webdataset_raw_pipeline(
                tar_file_path,
                index_file.name,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            webdataset_raw_pipeline(
                tar_file_path,
                index_file.name,
                ext=["Jpg", "cls"],
                case_sensitive_extensions=False,
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            test_batch_size,
            math.ceil(num_samples / num_shards / test_batch_size) * 2,
        )


def test_index_generation():
    num_samples = 3000
    tar_file_paths = [
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
    ]

    extract_dirs = [generate_temp_extract(tar_file_path) for tar_file_path in tar_file_paths]
    equivalent_files = sum(
        list(
            sorted(
                glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
            )  # noqa: 203
            for extract_dir in extract_dirs
        ),
        [],
    )

    num_shards = 100
    for shard_id in range(num_shards):
        compare_pipelines(
            webdataset_raw_pipeline(
                tar_file_paths,
                [],
                ["jpg", "cls"],
                missing_component_behavior="error",
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            file_reader_pipeline(
                equivalent_files,
                ["jpg", "cls"],
                num_shards=num_shards,
                shard_id=shard_id,
                batch_size=test_batch_size,
                device_id=0,
                num_threads=1,
            ),
            test_batch_size,
            math.ceil(num_samples / num_shards / test_batch_size) * 2,
        )
