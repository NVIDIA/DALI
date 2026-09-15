# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

import gcs_test_utils as gcs
from nose_utils import assert_raises, attr
from test_utils import compare_pipelines, to_array

batch_size = 4
num_threads = 4

g_server = None
g_tmpdir = None
g_root = None
g_files = None
g_tar = None
g_index = None
g_endpoint = None
g_quirks_files = None
g_odd_sizes = None
g_reserved_files = None
g_numpy_root = None

DATA_PREFIX = f"{gcs.PREFIX}/data"
WDS_PREFIX = f"{gcs.PREFIX}/wds"
MANY_PREFIX = f"{gcs.PREFIX}/many"
# Kept apart from DATA_PREFIX on purpose: the objects below are the ones discovery must throw
# away, and mixing them into the comparison fixture would make a failure hard to read.
QUIRKS_PREFIX = f"{gcs.PREFIX}/quirks"
ODD_PREFIX = f"{gcs.PREFIX}/odd"
# Object names holding characters that are reserved in a URI but ordinary in a GCS name.
RESERVED_PREFIX = f"{gcs.PREFIX}/reserved"
NUMPY_PREFIX = f"{gcs.PREFIX}/numpy"

# Sizes that are deliberately not round: they catch an off-by-one between the right-open
# ReadRange([begin, end)) that DALI issues and the inclusive HTTP "Range: bytes=first-last".
ODD_SIZES = (1, 4095, 4096, 4097, 65537)


def _make_local_dataset(root):
    files = []
    for cls in ("class_a", "class_b"):
        os.makedirs(os.path.join(root, cls))
        for i in range(4):
            name = os.path.join(cls, f"{i} x.dat")  # space on purpose: NVIDIA/DALI#5521
            with open(os.path.join(root, name), "w") as f:
                f.write(f"Contents of {name}.\n")
            files.append(name)
    empty = os.path.join("class_a", "empty.dat")  # 0-byte -> GetObjectMetadata branch
    open(os.path.join(root, empty), "w").close()
    files.append(empty)
    return sorted(files)


def _make_local_tar(path, num_samples=8):
    with tarfile.open(path, "w") as tar:
        for i in range(num_samples):
            for ext, payload in (("txt", b"sample-%d" % i), ("cls", b"%d" % i)):
                info = tarfile.TarInfo(f"{i:04}.{ext}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


def _seed_quirks(endpoint):
    """Seeds the objects that listing must reject, plus the two that must survive."""
    # directory markers - zero-byte objects whose *name* ends with '/'
    gcs.put_directory_marker(endpoint, f"{QUIRKS_PREFIX}/")
    gcs.put_directory_marker(endpoint, f"{QUIRKS_PREFIX}/class_a/")
    # depth 1: directly under the listed prefix, no subdirectory to take a label from
    gcs.put_object(endpoint, f"{QUIRKS_PREFIX}/toplevel.dat", b"toplevel")
    # depth 3: two levels below the listed prefix
    gcs.put_object(endpoint, f"{QUIRKS_PREFIX}/class_a/sub/deep.dat", b"deep")
    # the only two that are at exactly one subdirectory level
    kept = ["class_a/keep0.dat", "class_a/keep1.dat"]
    for name in kept:
        gcs.put_object(endpoint, f"{QUIRKS_PREFIX}/{name}", f"Contents of {name}.\n".encode())
    return sorted(kept)


def _seed_reserved_chars(endpoint):
    """Seeds names containing '?' and '#', which are reserved in a URI but legal in a GCS name.

    Discovery alone does not catch a parser that truncates at them: listing reports the name in
    full, so the object still becomes a sample and only the read fails. The test therefore has to
    compare contents, not just the epoch size.
    """
    names = ["class_a/plain.dat", "class_a/quest?mark.dat", "class_a/hash#mark.dat"]
    for name in names:
        gcs.put_object(endpoint, f"{RESERVED_PREFIX}/{name}", f"Contents of {name}.\n".encode())
    return sorted(names)


def _seed_numpy(endpoint, root):
    """Mirrors .npy files to GCS and to a local directory, in both layouts the reader accepts.

    readers.numpy reaches discovery through FileLoader with label_from_subdir off, so objects
    sitting directly under the prefix are samples as well - the flat layout is where GCS discovery
    can diverge from the local backend, which visits "." on top of the subdirectories.
    """
    for rel in ("nested/class_a", "flat"):
        os.makedirs(os.path.join(root, rel))
        for i in range(3):
            np.save(os.path.join(root, rel, f"{i:03}.npy"), np.full((2, 3), i, dtype=np.int32))
    gcs.upload_dir(endpoint, root, NUMPY_PREFIX)


def _seed_odd_sizes(endpoint, root):
    """Writes the same odd-sized payloads to GCS and to a local directory, for a byte compare."""
    os.makedirs(os.path.join(root, "odd"))
    for size in ODD_SIZES:
        payload = bytes((i * 7 + 11) % 256 for i in range(size))
        name = f"odd/{size:06}.dat"
        with open(os.path.join(root, name), "wb") as f:
            f.write(payload)
        gcs.put_object(endpoint, f"{ODD_PREFIX}/{name}", payload)


def _seed_many_objects(endpoint, count=1100):
    def put(i):
        gcs.put_object(endpoint, f"{MANY_PREFIX}/c/{i:04}.dat", b"x")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(put, range(count)))


def setUpModule():
    global g_server, g_tmpdir, g_root, g_files, g_tar, g_index, g_endpoint
    global g_quirks_files, g_odd_sizes, g_reserved_files, g_numpy_root
    gcs.require_mock_server()

    g_tmpdir = tempfile.TemporaryDirectory()
    g_root = os.path.join(g_tmpdir.name, "data")
    os.makedirs(g_root)
    g_files = _make_local_dataset(g_root)
    g_tar = os.path.join(g_tmpdir.name, "shard0.tar")
    _make_local_tar(g_tar)

    g_server = gcs.make_server()
    try:
        g_endpoint = g_server.start()
        gcs.export_gcs_env(g_endpoint)
        gcs.skip_if_no_gcs_support()
        gcs.create_bucket(g_endpoint)
        gcs.upload_dir(g_endpoint, g_root, DATA_PREFIX)
        with open(g_tar, "rb") as f:
            gcs.put_object(g_endpoint, f"{WDS_PREFIX}/shard0.tar", f.read())
        g_quirks_files = _seed_quirks(g_endpoint)
        g_reserved_files = _seed_reserved_chars(g_endpoint)
        g_numpy_root = os.path.join(g_tmpdir.name, "numpy")
        _seed_numpy(g_endpoint, g_numpy_root)
        g_odd_sizes = os.path.join(g_tmpdir.name, "sizes")
        _seed_odd_sizes(g_endpoint, g_odd_sizes)
    except Exception:
        tearDownModule()
        raise

    import webdataset_base as base

    g_index = base.generate_temp_index_file(g_tar)


def tearDownModule():
    global g_server, g_tmpdir, g_index, g_endpoint
    try:
        # The emulator is thrown away wholesale, but an external endpoint outlives the test run,
        # so everything uploaded there has to be removed again.
        if g_endpoint is not None and isinstance(g_server, gcs.ExternalGCSServer):
            gcs.delete_prefix(g_endpoint, gcs.PREFIX + "/")
        g_endpoint = None
        if g_server is not None:
            g_server.stop()
            g_server = None
    finally:
        if g_index is not None:
            g_index.close()
            g_index = None
        if g_tmpdir is not None:
            g_tmpdir.cleanup()
            g_tmpdir = None


@pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=None)
def file_pipe(file_root=None, files=None, dont_use_mmap=False):
    return tuple(
        fn.readers.file(
            file_root=file_root,
            files=files,
            name="Reader",
            dont_use_mmap=dont_use_mmap,
            file_filters=["*.dat"],
        )
    )


@pipeline_def(batch_size=3, num_threads=num_threads, device_id=None)
def numpy_pipe(file_root, dont_use_mmap=False):
    return fn.readers.numpy(file_root=file_root, name="Reader", dont_use_mmap=dont_use_mmap)


@pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=None)
def wds_pipe(paths, index_paths=None, dont_use_mmap=False):
    return tuple(
        fn.readers.webdataset(
            paths=paths, index_paths=index_paths, ext=["txt", "cls"], dont_use_mmap=dont_use_mmap
        )
    )


def _epoch_size(pipe):
    return pipe.reader_meta("Reader")["epoch_size"]


# ---------------------------------------------------------------------------------------------
# The same ground covered by the S3 suite
# ---------------------------------------------------------------------------------------------


def test_file_reader_file_root():
    compare_pipelines(
        file_pipe(file_root=f"gs://{gcs.BUCKET}/{DATA_PREFIX}"),
        file_pipe(file_root=g_root, dont_use_mmap=True),  # mmap cannot map the 0-byte file
        batch_size,
        3,
    )


def test_file_reader_files_arg():
    compare_pipelines(
        file_pipe(files=[f"gs://{gcs.BUCKET}/{DATA_PREFIX}/{f}" for f in g_files]),
        file_pipe(files=[os.path.join(g_root, f) for f in g_files], dont_use_mmap=True),
        batch_size,
        2,
    )


@attr("slow")
def test_file_reader_listing_pagination():
    # seeded here rather than in setUpModule: nothing else needs 1100 objects
    _seed_many_objects(g_endpoint)
    pipe = file_pipe(file_root=f"gs://{gcs.BUCKET}/{MANY_PREFIX}")
    pipe.build()
    assert _epoch_size(pipe) == 1100, pipe.reader_meta("Reader")


def test_webdataset_index_inferred():
    compare_pipelines(
        wds_pipe(paths=f"gs://{gcs.BUCKET}/{WDS_PREFIX}/shard0.tar"),
        wds_pipe(paths=g_tar, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_webdataset_local_index():
    compare_pipelines(
        wds_pipe(paths=f"gs://{gcs.BUCKET}/{WDS_PREFIX}/shard0.tar", index_paths=[g_index.name]),
        wds_pipe(paths=g_tar, index_paths=[g_index.name], dont_use_mmap=True),
        batch_size,
        2,
    )


def test_file_reader_missing_object():
    with assert_raises(RuntimeError, glob="*GCS object not found. bucket=*object=*"):
        pipe = file_pipe(files=[f"gs://{gcs.BUCKET}/{DATA_PREFIX}/no-such-object.dat"])
        pipe.build()
        pipe.run()


def test_file_reader_missing_bucket():
    with assert_raises(RuntimeError, glob="*Failed to list GCS objects*"):
        file_pipe(file_root="gs://dali-no-such-bucket/data").build()


# ---------------------------------------------------------------------------------------------
# GCS-specific: each of these pins a branch of gcs_discover_files / gcs_filesystem
# ---------------------------------------------------------------------------------------------


def test_directory_markers_are_not_samples():
    """A zero-byte object named "<prefix>/class_a/" is a folder, not a sample.

    lexically_relative() turns it into ("class_a", ""), so it only stays out of the dataset
    because of the empty-filename guard in gcs_discover_files.
    """
    pipe = file_pipe(file_root=f"gs://{gcs.BUCKET}/{QUIRKS_PREFIX}")
    pipe.build()
    assert _epoch_size(pipe) == len(g_quirks_files), pipe.reader_meta("Reader")


def test_listing_depth_filtering():
    """Only objects exactly one subdirectory below the prefix become samples.

    The fixture also holds an object directly under the prefix and one two levels down; both are
    rejected by the path_elems != 2 check, which is also what keeps the prefix's own directory
    marker (lexically_relative() maps it to ".") from being dereferenced.
    """
    pipe = file_pipe(file_root=f"gs://{gcs.BUCKET}/{QUIRKS_PREFIX}")
    pipe.build()
    data, _ = pipe.run()  # fn.readers.file yields (contents, label)
    read = sorted(bytes(to_array(data[i])).decode() for i in range(len(g_quirks_files)))
    assert read == [f"Contents of {name}.\n" for name in g_quirks_files], read


def test_reserved_uri_characters_in_object_names():
    """'?' and '#' are ordinary characters in a GCS object name.

    Neither starts a query or a fragment in a gs:// URI, so parse_uri must keep the whole name.
    Truncating at them is the worst kind of failure: the object is listed, becomes a sample, and
    only the read of the shortened name fails - so this asserts on the contents.
    """
    pipe = file_pipe(file_root=f"gs://{gcs.BUCKET}/{RESERVED_PREFIX}")
    pipe.build()
    assert _epoch_size(pipe) == len(g_reserved_files), pipe.reader_meta("Reader")
    data, _ = pipe.run()
    read = sorted(bytes(to_array(data[i])).decode() for i in range(len(g_reserved_files)))
    assert read == [f"Contents of {name}.\n" for name in g_reserved_files], read


def test_numpy_reader_nested_prefix():
    """readers.numpy over gs://, one subdirectory below the prefix."""
    compare_pipelines(
        numpy_pipe(file_root=f"gs://{gcs.BUCKET}/{NUMPY_PREFIX}/nested"),
        numpy_pipe(file_root=os.path.join(g_numpy_root, "nested"), dont_use_mmap=True),
        3,
        2,
    )


def test_numpy_reader_flat_prefix():
    """readers.numpy over gs://, objects directly under the prefix.

    Without label_from_subdir there is no subdirectory to take a label from, so these count as
    samples - the local backend reads them and discovery over gs:// has to agree. Dropping them
    surfaces as "No files found." rather than a wrong result, which is why this compares against
    the local reader instead of asserting a count.
    """
    compare_pipelines(
        numpy_pipe(file_root=f"gs://{gcs.BUCKET}/{NUMPY_PREFIX}/flat"),
        numpy_pipe(file_root=os.path.join(g_numpy_root, "flat"), dont_use_mmap=True),
        3,
        2,
    )


def test_read_size_not_multiple_of_chunk():
    """Byte-identical compare of objects whose size is not round.

    ReadRange is right-open while the HTTP byte range is inclusive; an off-by-one there drops or
    duplicates the last byte, which only an exact compare catches.
    """
    compare_pipelines(
        file_pipe(file_root=f"gs://{gcs.BUCKET}/{ODD_PREFIX}"),
        file_pipe(file_root=g_odd_sizes, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_concurrent_reads():
    """GCSClientManager hands out client copies because a single instance is not thread-safe."""
    gcs_pipe = file_pipe(
        file_root=f"gs://{gcs.BUCKET}/{DATA_PREFIX}", batch_size=len(g_files), num_threads=4
    )
    local_pipe = file_pipe(
        file_root=g_root, dont_use_mmap=True, batch_size=len(g_files), num_threads=4
    )
    compare_pipelines(gcs_pipe, local_pipe, len(g_files), 2)


_CHECKSUM_CHILD = r"""
import os, sys
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from test_utils import to_array

bucket, prefix, local_root = sys.argv[1], sys.argv[2], sys.argv[3]

@pipeline_def(batch_size=4, num_threads=2, device_id=None)
def pipe(file_root):
    return tuple(fn.readers.file(file_root=file_root, file_filters=["*.dat"], dont_use_mmap=True))

remote = pipe(file_root=f"gs://{bucket}/{prefix}")
remote.build()
local = pipe(file_root=local_root)
local.build()
for _ in range(2):
    remote_data, _ = remote.run()  # fn.readers.file yields (contents, label)
    local_data, _ = local.run()
    for i in range(4):
        np.testing.assert_array_equal(to_array(remote_data[i]), to_array(local_data[i]))
print("OK")
"""


def test_checksum_validation_enabled():
    """DALI_GCS_VERIFY_CHECKSUMS=1 must still return correct bytes for ranged reads.

    Runs in a subprocess: the client is a process-wide singleton built from an env snapshot, so
    the flag cannot be flipped once anything in this process has touched gs://.
    """
    env = dict(os.environ)
    env["DALI_GCS_VERIFY_CHECKSUMS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    child = subprocess.run(
        [sys.executable, "-c", _CHECKSUM_CHILD, gcs.BUCKET, DATA_PREFIX, g_root],
        env=env,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    assert "OK" in child.stdout, child.stdout
