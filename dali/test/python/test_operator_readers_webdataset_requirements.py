from test_operator_readers_webdataset_base import *


def test_single_ext_from_many():
    global test_batch_size
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
            "jpg",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, "jpg", batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_all_ext_from_many():
    global test_batch_size
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


def test_hidden():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/hidden.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(
        filter(
            lambda s: s.endswith(".txt"),
            sorted(glob(extract_dir.name + "/*")),
        )
    )

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
        math.ceil(num_samples / test_batch_size),
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


def test_non_files():
    global test_batch_size
    num_samples = 8
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/types_contents.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(
        filter(
            lambda s: s.endswith(".txt"),
            sorted(glob(extract_dir.name + "/*")),
        )
    )
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
        math.ceil(num_samples / test_batch_size),
    )


def test_return_empty():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/missing.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "txt"],
            missing_component_behavior="fillempty",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, ["jpg", []], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_skip_sample():
    global test_batch_size
    num_samples = 500
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/missing.tar"
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
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/missing.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )[:num_samples]

    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["jpg", "cls"],
        missing_component_behavior="raise",
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_raises(RuntimeError, wds_pipeline.build)


def test_different_components():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/scrambled.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "txt;cls"],
            missing_component_behavior="fillempty",
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
    global test_batch_size
    num_samples = 100
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/dtypes.tar"
    index_file = generate_temp_index_file(tar_file_path)

    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["float16", "int32", "float64"] * 2,
        dtypes=[
            dali.types.DALIDataType.FLOAT16,
            dali.types.DALIDataType.INT32,
            dali.types.DALIDataType.FLOAT64,
            dali.types.DALIDataType.INT8,
            dali.types.DALIDataType.INT8,
            dali.types.DALIDataType.INT8,
        ],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    for sample_idx in range(num_samples):
        if sample_idx % test_batch_size == 0:
            f16, i32, f64, f16b, i32b, f64b = wds_pipeline.run()
        assert (f16.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()
        assert (i32.as_array()[sample_idx % test_batch_size] == [int(sample_idx)] * 10).all()
        assert (f64.as_array()[sample_idx % test_batch_size] == [float(sample_idx)] * 10).all()


def test_wds_sharding():
    global test_batch_size
    num_samples = 3000
    tar_file_paths = [
        get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar",
        get_dali_extra_path() + "/db/webdataset/MNIST/devel-1.tar",
        get_dali_extra_path() + "/db/webdataset/MNIST/devel-2.tar",
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
    global test_batch_size
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
            "jpg;cls",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files, "cls", batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_common_files():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "cls"] * 10,
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        file_reader_pipeline(
            equivalent_files,
            ["jpg", "cls"] * 10,
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_sharding():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar"
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
