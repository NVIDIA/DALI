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


def corner_case(
    test_batch_size=test_batch_size, dtypes=None, missing_component_behavior="", **kwargs
):
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "cls"],
            missing_component_behavior=missing_component_behavior,
            dtypes=dtypes,
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs
        ),
        file_reader_pipeline(
            equivalent_files,
            ["jpg", "cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_batch_size_1():
    corner_case(test_batch_size=1)


def test_last_shard():
    num_shards = 100
    corner_case(num_shards=num_shards, shard_id=num_shards - 1)


def test_single_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            missing_component_behavior="skip",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    assert_equal(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_single_sample_and_junk():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single_junk.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    assert_equal(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_wide_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/wide.tar")
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    num_components = 1000
    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    assert_equal(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_mmap_dtype_incompatibility():
    assert_raises(
        RuntimeError,
        corner_case,
        dtypes=[dali.types.INT8, dali.types.FLOAT64],
        glob="has a size not divisible by the chosen dtype's size of",
    )


def test_dont_use_mmap():
    corner_case(dont_use_mmap=True)


def test_skip_cached_images():
    corner_case(skip_cached_images=True)


def test_pad_last_batch():
    corner_case(pad_last_batch=True)


def test_lazy_init():
    corner_case(lazy_init=True)


def test_read_ahead():
    corner_case(read_ahead=True)


def test_argument_errors():
    def uris_index_paths_error():
        webdataset_pipeline = webdataset_raw_pipeline(
            [
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
            ],
            [],
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        uris_index_paths_error,
        glob="Number of uris does not match the number of index files",
    )

    assert_raises(
        RuntimeError,
        corner_case,
        missing_component_behavior="SomethingInvalid",
        glob="Invalid value for missing_component_behavior",
    )
    corner_case(missing_component_behavior="Skip")

    assert_raises(
        RuntimeError,
        corner_case,
        dtypes=[dali.types.STRING, dali.types.STRING],
        glob="Unsupported output dtype. Supported types include",
    )
    assert_raises(
        RuntimeError,
        corner_case,
        dtypes=dali.types.INT8,
        glob="Number of extensions does not match the number of types",
    )


def test_index_errors():
    def less_components_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 100\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        less_components_error,
        glob="less components than stated at the beginning of the index file",
    )

    def no_extensions_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 1\n0\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, no_extensions_error, glob="no extensions provided for the sample")

    def corresponding_component_size_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 1\n0 jpg\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        corresponding_component_size_error,
        glob="size corresponding to the extension not found",
    )

    def smaller_final_offset_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 2\n0 jpg 0\n1024 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        smaller_final_offset_error,
        glob="reported final offset smaller than a sample start offset",
    )

    def offset_not_divisible_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n0 jpg 0\n1030 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/types_contents.tar"),
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, offset_not_divisible_error, glob="- offset * not divisible by")

    def final_offset_not_divisible_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3100 2\n0 jpg 0\n1024 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/types_contents.tar"),
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError, final_offset_not_divisible_error, glob="- final offset * not divisible by"
    )

    def offset_order_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n1024 png 0\n0 jpg 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/types_contents.tar"),
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, offset_order_error, glob="sample offsets not in order")

    def reported_component_sizes_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 1\n0 int32 40 float64 80 float16 40")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/dtypes.tar"),
            index_file.name,
            "float16",
            dtypes=dali.types.FLOAT64,
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()
        webdataset_pipeline.run()
        webdataset_pipeline.run()

    assert_raises(RuntimeError, reported_component_sizes_error, glob="reporting component sizes different to actual")

    def reported_extensions_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 1\n0 jpg 40 float64 80 float16 20")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/dtypes.tar"),
            index_file.name,
            "jpg",
            missing_component_behavior="skip",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()
        webdataset_pipeline.run()
        webdataset_pipeline.run()

    assert_raises(RuntimeError, reported_extensions_error, glob="reporting different extensions in a sample to the actual ones")

    def archive_too_short_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 1\n0 int32 40 float64 80 float16 20")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/empty.tar"),
            index_file.name,
            "jpg",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()
        webdataset_pipeline.run()
        webdataset_pipeline.run()

    assert_raises(RuntimeError, archive_too_short_error, glob="reporting a file longer than actual")
