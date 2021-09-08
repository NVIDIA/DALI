from test_operator_readers_webdataset_base import *


def corner_case(
    test_batch_size=test_batch_size, dtypes=None, missing_component_behavior="", **kwargs
):
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar"
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
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/single.tar"
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
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/single_junk.tar"
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
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/wide.tar"
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
    assert_raises(RuntimeError, corner_case, dtypes=[dali.types.INT8, dali.types.FLOAT64])


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
    def uris_configs_error():
        webdataset_pipeline = webdataset_raw_pipeline(
            [
                get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
                get_dali_extra_path() + "/db/webdataset/MNIST/devel-1.tar",
                get_dali_extra_path() + "/db/webdataset/MNIST/devel-2.tar",
            ],
            [],
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, uris_configs_error)

    assert_raises(RuntimeError, corner_case, missing_component_behavior="SomethingInvalid")
    corner_case(missing_component_behavior="Skip")

    assert_raises(RuntimeError, corner_case, dtypes=[dali.types.STRING, dali.types.STRING])
    assert_raises(RuntimeError, corner_case, dtypes=dali.types.INT8)


def test_index_errors():
    def less_components_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 100\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, less_components_error)

    def no_extensions_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 1\n0\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, no_extensions_error)

    def corresponding_component_size_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 1\n0 jpg\n")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, corresponding_component_size_error)

    def earlier_final_offset_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"512 2\n0 jpg 0\n1024 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
            index_file.name,
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, earlier_final_offset_error)

    def offset_not_divisible_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n0 jpg 0\n1030 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar",
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, offset_not_divisible_error)

    def final_offset_not_divisible_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3100 2\n0 jpg 0\n1024 png 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar",
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, offset_not_divisible_error)

    def offset_order_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n1024 png 0\n0 jpg 512")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar",
            index_file.name,
            "float16",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(RuntimeError, offset_order_error)

    def reported_component_sizes_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n0 int32 40 float64 80 float16 20")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar",
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
    assert_raises(RuntimeError, reported_component_sizes_error)

    def reported_extensions_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n0 jpg 40 float64 80 float16 20")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar",
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
    assert_raises(RuntimeError, reported_component_sizes_error)

    def archive_too_short_error():
        index_file = tempfile.NamedTemporaryFile()
        index_file.write(b"3072 2\n0 int32 40 float64 80 float16 20")
        index_file.flush()
        webdataset_pipeline = webdataset_raw_pipeline(
            get_dali_extra_path() + "/db/webdataset/sample-tar/empty.tar",
            index_file.name,
            "jpg",
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()
        webdataset_pipeline.run()
        webdataset_pipeline.run()

    assert_raises(RuntimeError, archive_too_short_error)
