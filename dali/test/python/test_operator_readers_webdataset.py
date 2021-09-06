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
    uris, configs, ext, missing_component_behavior="fillempty", dtype=None, dont_use_mmap=False
):
    out = readers.webdataset(
        uris=uris,
        configs=configs,
        ext=ext,
        missing_component_behavior=missing_component_behavior,
        dtype=dtype,
        dont_use_mmap=dont_use_mmap,
        prefetch_queue_depth=1,
    )
    return out if type(out) != list else tuple(out)


def filter_ext(files, exts):
    if type(exts) == str:
        exts = {exts}
    return list(filter(lambda s: any(map(lambda ext: s.endswith(ext), exts)), files))


@pipeline_def()
def file_reader_pipeline(files, exts=None):
    if type(exts) != list:
        exts = [exts]

    return tuple(
        readers.file(files=filter_ext(files, ext))[0] if type(ext) in {str, set} else ext
        for ext in exts
    )


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
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

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
        filter(lambda s: s.endswith(".txt"), sorted(glob(extract_dir.name + "/*")))
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
    assert_equal(wds_pipeline.epoch_size().values()[0] == num_samples)


def test_non_files():
    global test_batch_size
    num_samples = 8
    tar_file_path = get_dali_extra_path() + "/db/webdataset/sample-tar/types_components.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = list(
        filter(lambda s: s.endswith(".txt"), sorted(glob(extract_dir.name + "/*")))
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
    # only the first 500 samples are full, the rest only has .jpg
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/missing.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

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
    equivalent_files = sorted(glob(extract_dir.name + "/*"))[:num_samples]

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
    assert_equal(wds_pipeline.epoch_size().values()[0] == num_samples)


def test_raise_error_on_missing():
    global test_batch_size
    num_samples = 1000
    tar_file_path = get_dali_extra_path() + "/db/webdataset/MNIST/missing.tar"
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))[:num_samples]

    wds_pipeline = webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["jpg", "cls"],
        missing_component_behavior="raise",
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_raises(wds_pipeline.build())


def test_different_components():
    global test_batch_size
    num_samples = 1000
    tar_file_path = (
        get_dali_extra_path() + "/db/webdataset/MNIST/scrambled.tar"
    )  # has some of the .cls files changed up for .txt files
    index_file = generate_temp_index_file(tar_file_path)

    extract_dir = generate_temp_extract(tar_file_path)
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

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
        ["float16", "int32", "float64"],
        dtype=[
            dali.types.DALIDataType.FLOAT16,
            dali.types.DALIDataType.INT32,
            dali.types.DALIDataType.FLOAT64,
        ],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    wds_pipeline.build()
    for sample_idx in range(num_samples):
        f16, i32, f64 = wds_pipeline.run()
        for batch_idx in range(test_batch_size):
            assert all(f16.as_array() == [float(sample_idx * test_batch_size + batch_idx)] * 10)
            assert all(i32.as_array() == [int(sample_idx * test_batch_size + batch_idx)] * 10)
            assert all(f64.as_array() == [float(sample_idx * test_batch_size + batch_idx)] * 10)


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
        list(sorted(glob(extract_dir.name + "/*")) for extract_dir in extract_dirs), []
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
    equivalent_files = sorted(glob(extract_dir.name + "/*"))

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