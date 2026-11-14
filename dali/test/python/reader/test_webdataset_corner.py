# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import math
import nvidia.dali as dali
import os
import tarfile
import tempfile
from glob import glob
from nose_utils import assert_equals, assert_raises
from nose2.tools import params

import webdataset_base as base
from test_utils import compare_pipelines, get_dali_extra_path


def general_corner_case(
    test_batch_size=base.test_batch_size, dtypes=None, missing_component_behavior="", **kwargs
):
    num_samples = 1000
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = sorted(
        glob(extract_dir.name + "/*"), key=lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    )

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["jpg", "cls"],
            missing_component_behavior=missing_component_behavior,
            dtypes=dtypes,
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs,
        ),
        base.file_reader_pipeline(
            equivalent_files,
            ["jpg", "cls"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
            **kwargs,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size),
    )


def test_mmap_dtype_incompatibility():
    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=[dali.types.INT8, dali.types.FLOAT64],
        glob="component size and dtype incompatible",
    )


def test_lazy_init():
    general_corner_case(lazy_init=True)


def test_read_ahead():
    general_corner_case(read_ahead=True)


def test_single_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            missing_component_behavior="skip",
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_single_sample_and_junk():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/single_junk.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            ["txt"],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files, ["txt"], batch_size=test_batch_size, device_id=0, num_threads=1
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_wide_sample():
    test_batch_size = 1
    num_samples = 1
    tar_file_path = os.path.join(get_dali_extra_path(), "db/webdataset/sample-tar/wide.tar")
    index_file = base.generate_temp_index_file(tar_file_path)

    extract_dir = base.generate_temp_extract(tar_file_path)
    equivalent_files = list(sorted(glob(extract_dir.name + "/*")))

    num_components = 1000
    compare_pipelines(
        base.webdataset_raw_pipeline(
            tar_file_path,
            index_file.name,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        base.file_reader_pipeline(
            equivalent_files,
            [str(x) for x in range(num_components)],
            batch_size=test_batch_size,
            device_id=0,
            num_threads=1,
        ),
        test_batch_size,
        math.ceil(num_samples / test_batch_size) * 10,
    )
    wds_pipeline = base.webdataset_raw_pipeline(
        tar_file_path,
        index_file.name,
        ["txt"],
        batch_size=test_batch_size,
        device_id=0,
        num_threads=1,
    )
    assert_equals(list(wds_pipeline.epoch_size().values())[0], num_samples)


def test_argument_errors():
    def paths_index_paths_error():
        webdataset_pipeline = base.webdataset_raw_pipeline(
            [
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
                os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
            ],
            ["test.idx"],
            ["jpg", "cls"],
            batch_size=1,
            device_id=0,
            num_threads=1,
        )
        webdataset_pipeline.build()

    assert_raises(
        RuntimeError,
        paths_index_paths_error,
        glob="The number of index files, if any, must match the number of archives in the dataset",
    )

    assert_raises(
        RuntimeError,
        general_corner_case,
        missing_component_behavior="SomethingInvalid",
        glob="Invalid value for missing_component_behavior",
    )
    general_corner_case(missing_component_behavior="Skip")

    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=[dali.types.STRING, dali.types.STRING],
        glob="Unsupported output dtype *. Supported types are",
    )
    assert_raises(
        RuntimeError,
        general_corner_case,
        dtypes=dali.types.INT8,
        glob="Number of extensions does not match the number of provided types",
    )


def general_index_error(
    index_file_contents, tar_file_path="db/webdataset/MNIST/devel-0.tar", ext="jpg"
):
    index_file = tempfile.NamedTemporaryFile()
    index_file.write(index_file_contents)
    index_file.flush()
    webdataset_pipeline = base.webdataset_raw_pipeline(
        os.path.join(get_dali_extra_path(), tar_file_path),
        index_file.name,
        ext,
        batch_size=1,
        device_id=0,
        num_threads=1,
    )
    webdataset_pipeline.run()
    webdataset_pipeline.run()


def test_index_errors():
    assert_raises(RuntimeError, general_index_error, b"", glob="no version signature found")
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v0.1",
        glob="Unsupported version of the index file (v0.1).",
    )
    assert_raises(RuntimeError, general_index_error, b"v1.1", glob="no sample count found")
    assert_raises(
        RuntimeError, general_index_error, b"v1.1 -1", glob="sample count must be positive"
    )
    assert_raises(
        RuntimeError, general_index_error, b"v1.1 1\n", glob="no extensions provided for the sample"
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg",
        glob="Could not find all necessary component parameters",
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg 1 0",
        glob="tar offset is not a multiple of tar block size",
    )
    assert_raises(
        RuntimeError,
        general_index_error,
        b"v1.1 1\njpg 51200 1",
        "db/webdataset/sample-tar/empty.tar",
        glob="offset or size is outside of the archive file",
    )


def run_component_range_reader(archive, index, use_mmap, dtype=dali.types.UINT8):
    @dali.pipeline_def(batch_size=1, num_threads=1, device_id=None, prefetch_queue_depth=1)
    def pipeline():
        return dali.fn.readers.webdataset(
            paths=[archive],
            index_paths=[index],
            ext=["bin"],
            dtypes=[dtype],
            dont_use_mmap=not use_mmap,
            random_shuffle=False,
            initial_fill=1,
        )

    return pipeline().run()[0].as_array()


@params(
    *[
        (mmap, offset, size)
        for mmap in (False, True)
        for offset, size in (
            (512, 2**64 - 4),
            (512, 2**64 - 512),
            (-512, 4),
            (512, 513),
            (512, 1028),
            (2048, 4),
        )
    ]
)
def test_invalid_component_range(use_mmap, offset, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(1024))
        with open(index, "w") as f:
            f.write(f"v1.2 1\nbin {offset} {size} sample.bin\n")
        # Keep unaligned sizes compatible with the dtype so they reach the range check.
        dtype = dali.types.FLOAT if size % 4 == 0 else dali.types.UINT8
        with assert_raises(RuntimeError, glob="*offset or size is outside*"):
            run_component_range_reader(archive, index, use_mmap, dtype)


@params(*[(use_mmap, size) for use_mmap in (False, True) for size in (0, 4)])
def test_component_ending_at_eof(use_mmap, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(512) + bytes(range(size)))
        with open(index, "w") as f:
            f.write(f"v1.2 1\nbin 512 {size} sample.bin\n")
        assert run_component_range_reader(archive, index, use_mmap).tolist() == [list(range(size))]


def run_tar_size_reader(archive, use_mmap, dtype=dali.types.UINT8):
    @dali.pipeline_def(batch_size=1, num_threads=1, device_id=None, prefetch_queue_depth=1)
    def pipeline():
        return dali.fn.readers.webdataset(
            paths=[archive],
            index_paths=[],
            ext=["bin"],
            dtypes=[dtype],
            dont_use_mmap=not use_mmap,
            random_shuffle=False,
            initial_fill=1,
        )

    return pipeline().run()[0].as_array()


@params(False, True)
def test_invalid_mapping(use_mmap):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(1024))
        with open(index, "w") as f:
            f.write("v1.2 1\nbin 512 1024 sample.bin\n")
        message = "*offset or size is outside of the archive file*"
        with assert_raises(RuntimeError, glob=message):
            run_component_range_reader(archive, index, use_mmap)


def write_tar_size_field(archive, field):
    assert len(field) == 12
    header = bytearray(tarfile.TarInfo("sample.bin").tobuf())
    header[124:136] = field
    header[148:156] = b" " * 8
    header[148:156] = f"{sum(header):06o}\0 ".encode()
    with open(archive, "wb") as f:
        f.write(header + bytes(1024))


def write_invalid_tar(archive, size):
    field = oct(size)[2:].encode().rjust(11, b" ") + b"\0"
    write_tar_size_field(archive, field)


@params(
    *[
        (use_mmap, size)
        for use_mmap in (False, True)
        for size in (2**32, 2**32 + 4, 8**11 - 1, 2048)
    ]
)
def test_invalid_tar_entry_size(use_mmap, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        write_invalid_tar(archive, size)
        message = (
            f"*{archive}: entry sample.bin declares {size} bytes, "
            "but only 1024 bytes are available at offset 512; "
            "the entry extends beyond the archive.*"
        )
        with assert_raises(RuntimeError, glob=message):
            run_tar_size_reader(archive, use_mmap, dali.types.FLOAT)


@params(
    *[
        (use_mmap, field)
        for use_mmap in (False, True)
        for field in (
            b"         -4\0",  # Negative octal.
            b"00000000008\0",  # Invalid octal digit.
            b"0000000004x\0",  # Trailing garbage after valid digits.
            b"0004" + b"\0" + b"0000000",  # Digits after a terminator.
            b" " * 12,  # Missing digits.
            b"\xff" * 12,  # Negative GNU base-256.
            b"\x80" + (2**64).to_bytes(11, "big"),  # Base-256 overflow.
        )
    ]
)
def test_malformed_tar_entry_size(use_mmap, field):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        write_tar_size_field(archive, field)
        with assert_raises(RuntimeError, glob="*entry sample.bin: invalid tar entry size.*"):
            run_tar_size_reader(archive, use_mmap)


@params(*[(use_mmap, size) for use_mmap in (False, True) for size in (2**32 + 4, 2**64 - 1)])
def test_oversized_base256_tar_entry(use_mmap, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        write_tar_size_field(archive, b"\x80" + size.to_bytes(11, "big"))
        with assert_raises(
            RuntimeError, glob=f"*declares {size} bytes*extends beyond the archive.*"
        ):
            run_tar_size_reader(archive, use_mmap)


@params(
    *[
        (use_mmap, field, size)
        for use_mmap in (False, True)
        for field, size in (
            (b"000000000004", 4),  # Full-width octal without a terminator.
            (b" 000000004 \0", 4),  # Leading and trailing padding.
            (b"\x80" + (4).to_bytes(11, "big"), 4),
            (b"\x80" + bytes(11), 0),
        )
    ]
)
def test_valid_tar_size_encoding(use_mmap, field, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        write_tar_size_field(archive, field)
        assert run_tar_size_reader(archive, use_mmap).tolist() == [[0] * size]


@params(False, True)
def test_repeated_invalid_tar_first_entry(use_mmap):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "invalid.tar")
        write_invalid_tar(archive, 2048)
        for _ in range(32):
            with assert_raises(RuntimeError, glob="*extends beyond the archive.*"):
                run_tar_size_reader(archive, use_mmap)


@params(False, True)
def test_valid_tar(use_mmap):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("sample.bin")
            info.size = 4
            tar.addfile(info, io.BytesIO(bytes(range(4))))
        assert run_tar_size_reader(archive, use_mmap).tolist() == [[0, 1, 2, 3]]


@params(
    *[
        (use_mmap, kind)
        for use_mmap in (False, True)
        for kind in (
            tarfile.LNKTYPE,
            tarfile.SYMTYPE,
            tarfile.DIRTYPE,
            tarfile.CHRTYPE,
            tarfile.BLKTYPE,
            tarfile.FIFOTYPE,
        )
    ]
)
def test_tar_non_data_entry_ignores_size(use_mmap, kind):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        entry = tarfile.TarInfo("metadata")
        entry.type = kind
        entry.linkname = "sample.bin"
        entry.size = 8**11 - 1
        sample = tarfile.TarInfo("sample.bin")
        sample.size = 4
        # These entry types carry no data blocks, even when their size field is nonzero.
        with open(archive, "wb") as f:
            f.write(entry.tobuf())
            f.write(sample.tobuf())
            f.write(bytes(range(4)) + bytes(508))
            f.write(bytes(1024))
        assert run_tar_size_reader(archive, use_mmap).tolist() == [[0, 1, 2, 3]]
