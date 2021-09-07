import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.fn.readers as readers
from test_utils import compare_pipelines, get_dali_extra_path
from nose.tools import assert_raises, assert_equal
import tempfile
from subprocess import call
import os
from glob import glob
import tarfile
import numpy as np
import math

test_batch_size = 4
wds2idx_script = "../../../tools/wds2idx.py"


@pipeline_def()
def webdataset_raw_pipeline(
    uris,
    configs,
    ext,
    missing_component_behavior="fillempty",
    dtypes=None,
    dont_use_mmap=False,
    num_shards=1,
    shard_id=0,
    skip_cached_images=False,
    pad_last_batch=False,
    lazy_init=False,
    read_ahead=False,
    stick_to_shard=False
):
    out = readers.webdataset(
        uris=uris,
        configs=configs,
        ext=ext,
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
    return out if type(out) != list else tuple(out)


def filter_ext(files, exts):
    if type(exts) == str:
        exts = {exts}
    return list(filter(lambda s: any(map(lambda ext: s.endswith(ext), exts)), files))


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
    stick_to_shard=False
):
    if type(exts) != list:
        exts = [exts]

    return tuple(
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
        for ext in exts
    )


def generate_temp_index_file(tar_file_path):
    global wds2idx_script
    temp_index_file = tempfile.NamedTemporaryFile()
    call([wds2idx_script, tar_file_path, temp_index_file.name], stdout=open(os.devnull, "wb"))
    return temp_index_file


def generate_temp_extract(tar_file_path):
    temp_extract_dir = tempfile.TemporaryDirectory()
    archive = tarfile.open(tar_file_path)
    for member in archive:
        if member.type != tarfile.REGTYPE:
            continue
        archive.extract(member, temp_extract_dir.name)
    return temp_extract_dir
