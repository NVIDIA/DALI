# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from test_operator_readers_webdataset_base import *


def requirements_general(
    num_samples,
    tar_file_path,
    extensions,
    batch_size=test_batch_size,
    sort_by_index=False,
    file_extensions=None,
    missing_component_behavior="",
    **kwargs
):
    if file_extensions is None:
        file_extensions = extensions
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = glob(extract_dir.name + "/*")
    equivalent_files = sorted(
        equivalent_files,
        **({"key": (lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")]))} if sort_by_index else {})
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            extensions,
            batch_size=batch_size,
            device_id=0,
            num_threads=1,
            missing_component_behavior=missing_component_behavior,
            **kwargs
        ),
        file_reader_pipeline(
            equivalent_files,
            file_extensions,
            batch_size=batch_size,
            device_id=0,
            num_threads=1,
            **kwargs
        ),
        batch_size,
        math.ceil(num_samples / batch_size),
    )

    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        extensions,
        batch_size=batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    assert_equal(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_single_ext_from_many():
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        "jpg",
        sort_by_index=True,
    )


def test_all_ext_from_many():
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        ["jpg", "cls"],
        sort_by_index=True,
    )


def test_hidden():
    requirements_general(
        1,
        os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/hidden.tar"),
        "txt",
        batch_size=1,
    )


def test_non_files():
    requirements_general(
        8, os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/types_contents.tar"), "txt"
    )


def test_return_empty():
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar"),
        ["jpg", "txt"],
        sort_by_index=True,
        missing_component_behavior="empty",
        file_extensions=["jpg", []],
    )


def test_skip_sample():
    global test_batch_size
    num_samples = 500
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(
        filter(
            lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")]) < 2500,
            sorted(
                glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
            ),
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
    wds_pipeline.build()
    assert_equal(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_raise_error_on_missing():
    global test_batch_size
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/missing.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )[:num_samples]

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
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/scrambled.tar"),
        ["jpg", "txt;cls"],
        missing_component_behavior="empty",
        sort_by_index=True,
        file_extensions=["jpg", {"txt", "cls"}],
    )


def test_dtypes():
    global test_batch_size
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
    wds_pipeline.build()
    for sample_idx in range(num_samples):
        if sample_idx % test_batch_size == 0:
            f16, i32, f64 = wds_pipeline.run()
        assert (f16.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()
        assert (i32.as_array()[sample_idx % test_batch_size] == [int(sample_idx)] * 10).all()
        assert (f64.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()


def test_wds_sharding():
    global test_batch_size
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
            )
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


def test_ambiguous_components():
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        "jpg;cls",
        sort_by_index=True,
        file_extensions="cls",
    )


def test_common_files():
    requirements_general(
        1000,
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        ["jpg", "cls"] * 10,
        sort_by_index=True
    )


def test_sharding():
    global test_batch_size
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

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
