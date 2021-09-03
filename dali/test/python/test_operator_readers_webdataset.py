from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.fn.readers as readers
from test_utils import compare_pipelines, get_dali_extra_path
from nose.tools import assert_raises
import tempfile
from subprocess import call
import os
from glob import glob
import tarfile
import numpy as np
import math

test_batch_size = 16
wds2idx_script = "../../../tools/wds2idx.py"


@pipeline_def()
def webdataset_raw_pipeline(uris, configs, ext, missing_component_behavior="fillempty", dtype=None, dont_use_mmap=False):
    out = readers.webdataset(
        uris=uris, 
        configs=configs, 
        ext=ext, 
        missing_component_behavior=missing_component_behavior, 
        dtype=dtype, 
        dont_use_mmap=dont_use_mmap,
        prefetch_queue_depth=1
    )
    return out if type(out) != list else tuple(out)
@pipeline_def()
def file_reader_pipeline(files, ext):
    if type(ext) == str:
        ext = [ext]
    return tuple(readers.file(files=list(filter(lambda s: s.endswith(component_ext), files)))[0] for component_ext in ext)
    
def generate_temp_index_file(tar_file_path):
    global wds2idx_script
    temp_index_file = tempfile.NamedTemporaryFile()
    call([wds2idx_script, tar_file_path, temp_index_file.name])
    return temp_index_file

def generate_temp_extract(tar_file_path):
    temp_extract_dir = tempfile.TemporaryDirectory()
    archive = tarfile.open(tar_file_path)
    for member in archive:
        if member.type != tarfile.REGTYPE:
            continue
        archive.extract(member, temp_extract_dir.name)
    return temp_extract_dir

def test_single_ext_from_many():
    global test_batch_size
    num_samples = 1000

    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

    compare_pipelines(
        webdataset_raw_pipeline(tar_file_path, index_file.name, "jpg", 
                                       batch_size=test_batch_size, device_id=0, num_threads=1),
        file_reader_pipeline(equivalent_files, "jpg", batch_size=test_batch_size, device_id=0, num_threads=1),
        test_batch_size,
        math.ceil(num_samples / test_batch_size)
    )

def test_all_ext_from_many():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/devel-0.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

    compare_pipelines(
        webdataset_raw_pipeline(tar_file_path, index_file.name, ["jpg", "cls"],
                                       batch_size=test_batch_size, device_id=0, num_threads=1),
        file_reader_pipeline(equivalent_files, ["jpg", "cls"], batch_size=test_batch_size, device_id=0, num_threads=1),
        test_batch_size,
        math.ceil(num_samples / test_batch_size)
    )

def test_hidden():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/hidden.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(filter(lambda s: s.endswith('.txt'), sorted(glob(extract_dir.name + "/*"))))

    compare_pipelines(
        webdataset_raw_pipeline(tar_file_path, index_file.name, ["txt"],
                                       batch_size=test_batch_size, device_id=0, num_threads=1),
        file_reader_pipeline(equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1),
        test_batch_size,
        math.ceil(num_samples / test_batch_size)
    )

    wds_pipeline = webdataset_raw_pipeline(tar_file_path, index_file.name, ["txt"], 
                                        batch_size=test_batch_size, device_id=0, num_threads=1)
    wds_pipeline.build()
    assert_equal(wds_pipeline.epoch_size().values()[0] == 1)