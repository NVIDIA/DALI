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
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

import s3_test_utils as s3
from nose_utils import assert_raises, attr
from test_utils import compare_pipelines

batch_size = 4
num_threads = 4

g_server = None
g_tmpdir = None
g_root = None
g_files = None
g_tar = None
g_index = None


def _make_local_dataset(root):
    files = []
    for cls in ("class_a", "class_b"):
        os.makedirs(os.path.join(root, cls))
        for i in range(4):
            name = os.path.join(cls, f"{i} x.dat")  # space on purpose: NVIDIA/DALI#5521
            with open(os.path.join(root, name), "w") as f:
                f.write(f"Contents of {name}.\n")
            files.append(name)
    empty = os.path.join("class_a", "empty.dat")  # 0-byte -> HeadObject branch
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


def _seed_many_objects(client, count=1100):
    def put(i):
        client.put_object(Bucket=s3.BUCKET, Key=f"many/c/{i:04}.dat", Body=b"x")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(put, range(count)))


def setUpModule():
    global g_server, g_tmpdir, g_root, g_files, g_tar, g_index
    s3.skip_if_no_mock_server()

    g_tmpdir = tempfile.TemporaryDirectory()
    g_root = os.path.join(g_tmpdir.name, "data")
    os.makedirs(g_root)
    g_files = _make_local_dataset(g_root)
    g_tar = os.path.join(g_tmpdir.name, "shard0.tar")
    _make_local_tar(g_tar)

    g_server = s3.make_server()
    try:
        endpoint = g_server.start()
        s3.export_s3_env(endpoint)
        s3.skip_if_no_s3_support()
        client = s3.s3_client(endpoint)
        client.create_bucket(Bucket=s3.BUCKET)
        s3.upload_dir(client, s3.BUCKET, g_root, "data")
        client.upload_file(g_tar, s3.BUCKET, "wds/shard0.tar")
    except Exception:
        tearDownModule()
        raise

    import webdataset_base as base

    g_index = base.generate_temp_index_file(g_tar)


def tearDownModule():
    global g_server, g_tmpdir, g_index
    try:
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


@pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=None)
def wds_pipe(paths, index_paths=None, dont_use_mmap=False):
    return tuple(
        fn.readers.webdataset(
            paths=paths, index_paths=index_paths, ext=["txt", "cls"], dont_use_mmap=dont_use_mmap
        )
    )


def test_file_reader_file_root():
    compare_pipelines(
        file_pipe(file_root=f"s3://{s3.BUCKET}/data"),
        file_pipe(file_root=g_root, dont_use_mmap=True),  # mmap cannot map the 0-byte file
        batch_size,
        3,
    )


def test_file_reader_files_arg():
    compare_pipelines(
        file_pipe(files=[f"s3://{s3.BUCKET}/data/{f}" for f in g_files]),
        file_pipe(files=[os.path.join(g_root, f) for f in g_files], dont_use_mmap=True),
        batch_size,
        2,
    )


@attr("slow")
def test_file_reader_listing_pagination():
    # seeded here rather than in setUpModule: it costs ~7 s and nothing else needs it
    _seed_many_objects(s3.s3_client(g_server.endpoint_url))
    pipe = file_pipe(file_root=f"s3://{s3.BUCKET}/many")
    pipe.build()
    assert pipe.reader_meta("Reader")["epoch_size"] == 1100, pipe.reader_meta("Reader")


def test_webdataset_index_inferred():
    compare_pipelines(
        wds_pipe(paths=f"s3://{s3.BUCKET}/wds/shard0.tar"),
        wds_pipe(paths=g_tar, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_webdataset_local_index():
    compare_pipelines(
        wds_pipe(paths=f"s3://{s3.BUCKET}/wds/shard0.tar", index_paths=[g_index.name]),
        wds_pipe(paths=g_tar, index_paths=[g_index.name], dont_use_mmap=True),
        batch_size,
        2,
    )


def test_file_reader_missing_object():
    with assert_raises(RuntimeError, glob="*S3 Object not found. bucket=*object=*"):
        pipe = file_pipe(files=[f"s3://{s3.BUCKET}/data/no-such-object.dat"])
        pipe.build()
        pipe.run()


def test_file_reader_missing_bucket():
    with assert_raises(RuntimeError, glob="*NoSuchBucket*"):
        file_pipe(file_root="s3://dali-no-such-bucket/data").build()
