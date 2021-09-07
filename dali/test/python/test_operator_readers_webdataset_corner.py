from test_operator_readers_webdataset_base import *


def corner_case(
    test_batch_size=test_batch_size,
    **kwargs
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
    global test_batch_size
    num_shards = 100
    corner_case(num_shards=num_shards, shard_id=num_shards - 1)


def test_dont_use_mmap():
    corner_case(dont_use_mmap=True)


def test_mmap_dtype_incompatibility():
    pass


def test_skip_cached_images():
    corner_case(skip_cached_images=True)


def test_pad_last_batch():
    corner_case(pad_last_batch=True)


def test_lazy_init():
    corner_case(lazy_init=True)


def test_read_ahead():
    corner_case(read_ahead=True)


def test_errors():
    pass
