from test_operator_readers_webdataset_base import *

def cross_check(
    dont_use_mmap,
    batch_size,
    num_shards,
    shard_id,
    prefetch_queue_depth,
    skip_cached_images,
    pad_last_batch,
    lazy_init,
    read_ahead,
    stick_to_shard,
):
    num_samples = 100
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/cross.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(
        glob(extract_dir.name + "/*"),
        key=lambda s: (int(s[s.rfind("/") + 1 : s.find(".")]), s),
    ))

    compare_pipelines(
        webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["a.a;a.b", "b.a;b.b"],
            batch_size=batch_size,
            device_id=0,
            num_threads=10,
            dont_use_mmap=dont_use_mmap,
            num_shards=num_shards,
            shard_id=shard_id,
            prefetch_queue_depth=prefetch_queue_depth,
            skip_cached_images=skip_cached_images,
            pad_last_batch=pad_last_batch,
            lazy_init=lazy_init,
            read_ahead=read_ahead,
            stick_to_shard=stick_to_shard,
        ),
        file_reader_pipeline(
            equivalent_files,
            ["a.a", "b.a"],
            batch_size=batch_size,
            device_id=0,
            num_threads=10,
            dont_use_mmap=True,
            num_shards=num_shards,
            shard_id=shard_id,
            prefetch_queue_depth=prefetch_queue_depth,
            skip_cached_images=skip_cached_images,
            pad_last_batch=pad_last_batch,
            lazy_init=lazy_init,
            read_ahead=read_ahead,
            stick_to_shard=stick_to_shard,
        ),
        batch_size,
        math.ceil(num_samples / test_batch_size),
    )

def test_cross_check():
    scenarios = [
        (
            dont_use_mmap,
            batch_size,
            num_shards,
            shard_id,
            prefetch_queue_depth,
            skip_cached_images,
            pad_last_batch,
            lazy_init,
            read_ahead,
            stick_to_shard,
        )
        for dont_use_mmap in (False, True)
        for stick_to_shard in (False, True)
        for read_ahead in (False, True)
        for lazy_init in (False, True)
        for pad_last_batch in (False, True)
        for skip_cached_images in (False, True)
        for batch_size in (1, 8)
        for num_shards in (1, 100)
        for shard_id in {0, num_shards-1}
        for prefetch_queue_depth in (1, 8)
    ]

    for args in scenarios:
        yield (cross_check,) + args
