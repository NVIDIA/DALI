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
import struct
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

import s3_test_utils as s3
from nose2.tools import params
from nose_utils import assert_raises, attr
from test_utils import compare_pipelines, get_dali_extra_path, to_array

batch_size = 4
num_threads = 4

g_server = None
g_tmpdir = None
g_root = None
g_files = None
g_tar = None
g_index = None
g_rec = None
g_rec_index = None
g_client = None
g_quirks_files = None
g_numpy_root = None
g_video_root = None
g_bucket_root = None

DATA_PREFIX = f"{s3.PREFIX}/data"
WDS_PREFIX = f"{s3.PREFIX}/wds"
MANY_PREFIX = f"{s3.PREFIX}/many"
REC_PREFIX = f"{s3.PREFIX}/rec"
# Kept apart from DATA_PREFIX on purpose: the objects below are the ones discovery must throw
# away, and mixing them into the comparison fixture would make a failure hard to read.
QUIRKS_PREFIX = f"{s3.PREFIX}/quirks"
NUMPY_PREFIX = f"{s3.PREFIX}/numpy"
VIDEO_PREFIX = f"{s3.PREFIX}/video"
VIDEO_ROOT_PREFIX = f"{s3.PREFIX}/video_root"

# A bucket of its own, listed from the root: nothing else may be in it, since every object at
# the right depth becomes a sample.
ROOT_BUCKET = f"{s3.PREFIX}-root"

# Videos for each backend of the video reader, relative to DALI_extra/db/video - the CPU one
# decodes only VP8, VP9 and MJPEG. The last file of each is larger than the 1 MiB I/O buffer the
# reader uses for streamed videos, so decoding it takes several ranged reads, not just one.
VIDEO_FILES = {
    "cpu": ["vp9/vp9_0.mp4", "vp8/vp8.webm", "mjpeg/mjpeg.avi"],
    "gpu": ["cfr_test.mp4", "vfr/test_2.mp4", "sintel/video_files/sintel_trailer-720p_2.mp4"],
}


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


def _make_local_recordio(path, index_path, num_samples=8):
    """A minimal MXNet RecordIO shard: magic (0xced7230a) + length/cflag word (cflag=0, no
    continuation) + a 24-byte ImageRecordIOHeader (flag=0 -> a single float label, no extra
    labels) + the sample payload. The index is plain text, "<ignored> <byte offset>" per line -
    RecordIOLoader::ReadIndexFile only reads the second column (dali/operators/reader/loader/
    recordio_loader.h)."""
    with open(path, "wb") as rec, open(index_path, "w") as idx:
        offset = 0
        for i in range(num_samples):
            payload = struct.pack("<IfQQ", 0, float(i), 0, 0) + b"sample-%d" % i
            record = struct.pack("<II", 0xCED7230A, len(payload)) + payload
            rec.write(record)
            idx.write(f"{i} {offset}\n")
            offset += len(record)


def _seed_many_objects(client, count=1100):
    def put(i):
        client.put_object(Bucket=s3.BUCKET, Key=f"{MANY_PREFIX}/c/{i:04}.dat", Body=b"x")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(put, range(count)))


def _put(client, key, body):
    client.put_object(Bucket=s3.BUCKET, Key=key, Body=body)


def _seed_quirks(client):
    """Seeds the objects that listing must reject, plus the two that must survive."""
    # directory markers - zero-byte objects whose key ends with '/', as made by "Create folder"
    _put(client, f"{QUIRKS_PREFIX}/", b"")
    _put(client, f"{QUIRKS_PREFIX}/class_a/", b"")
    # depth 1: directly under the listed prefix, no subdirectory to take a label from
    _put(client, f"{QUIRKS_PREFIX}/toplevel.dat", b"toplevel")
    # depth 3: two levels below the listed prefix
    _put(client, f"{QUIRKS_PREFIX}/class_a/sub/deep.dat", b"deep")
    # the only two that are at exactly one subdirectory level
    kept = ["class_a/keep0.dat", "class_a/keep1.dat"]
    for name in kept:
        _put(client, f"{QUIRKS_PREFIX}/{name}", f"Contents of {name}.\n".encode())
    return sorted(kept)


def _seed_numpy(client, root):
    """Mirrors .npy files to S3 and to a local directory, in both layouts the reader accepts.

    readers.numpy reaches discovery through FileLoader with label_from_subdir off, so objects
    sitting directly under the prefix are samples as well - the flat layout is where S3 discovery
    can diverge from the local backend, which visits "." on top of the subdirectories.
    """
    for rel in ("nested/class_a", "flat"):
        os.makedirs(os.path.join(root, rel))
        for i in range(3):
            np.save(os.path.join(root, rel, f"{i:03}.npy"), np.full((2, 3), i, dtype=np.int32))
    s3.upload_dir(client, s3.BUCKET, root, NUMPY_PREFIX)


def _seed_bucket_root(client, root):
    """Mirrors a small file reader dataset to the root of ROOT_BUCKET and to a local directory.

    The object directly under the root must not become a sample - the local reader skips it too,
    as it has no subdirectory to take a label from.
    """
    for name in ("class_a/0.dat", "class_a/1.dat", "class_b/0.dat", "toplevel.dat"):
        os.makedirs(os.path.join(root, os.path.dirname(name)), exist_ok=True)
        with open(os.path.join(root, name), "w") as f:
            f.write(f"Contents of {name}.\n")
    s3.create_bucket(client, ROOT_BUCKET)
    s3.upload_dir(client, ROOT_BUCKET, root, "")


def _seed_videos(client):
    for device_files in VIDEO_FILES.values():
        for name in device_files:
            client.upload_file(_local_video(name), s3.BUCKET, f"{VIDEO_PREFIX}/{name}")


def _local_video(name):
    return os.path.join(get_dali_extra_path(), "db", "video", name)


def _remote_video(name):
    return f"s3://{s3.BUCKET}/{VIDEO_PREFIX}/{name}"


def _make_video_root(root):
    """A file_root for each backend: its videos split between two label subdirectories."""
    for device, files in VIDEO_FILES.items():
        for i, name in enumerate(files):
            path = os.path.join(root, device, "ab"[i % 2], os.path.basename(name))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.symlink(_local_video(name), path)


def setUpModule():
    global g_server, g_tmpdir, g_root, g_files, g_tar, g_index, g_rec, g_rec_index, g_client
    global g_quirks_files, g_numpy_root, g_bucket_root, g_video_root
    s3.require_mock_server()

    g_tmpdir = tempfile.TemporaryDirectory()
    g_root = os.path.join(g_tmpdir.name, "data")
    os.makedirs(g_root)
    g_files = _make_local_dataset(g_root)
    g_tar = os.path.join(g_tmpdir.name, "shard0.tar")
    _make_local_tar(g_tar)
    g_rec = os.path.join(g_tmpdir.name, "train.rec")
    g_rec_index = os.path.join(g_tmpdir.name, "train.idx")
    _make_local_recordio(g_rec, g_rec_index)

    g_server = s3.make_server()
    try:
        endpoint = g_server.start()
        s3.export_s3_env(endpoint)
        s3.require_s3_support()
        g_client = s3.s3_client(endpoint)
        s3.create_bucket(g_client, s3.BUCKET)
        s3.upload_dir(g_client, s3.BUCKET, g_root, DATA_PREFIX)
        g_client.upload_file(g_tar, s3.BUCKET, f"{WDS_PREFIX}/shard0.tar")
        g_client.upload_file(g_rec, s3.BUCKET, f"{REC_PREFIX}/train.rec")
        g_client.upload_file(g_rec_index, s3.BUCKET, f"{REC_PREFIX}/train.idx")
        g_quirks_files = _seed_quirks(g_client)
        g_numpy_root = os.path.join(g_tmpdir.name, "numpy")
        _seed_numpy(g_client, g_numpy_root)
        g_bucket_root = os.path.join(g_tmpdir.name, "bucket_root")
        _seed_bucket_root(g_client, g_bucket_root)
        _seed_videos(g_client)
        g_video_root = os.path.join(g_tmpdir.name, "video_root")
        _make_video_root(g_video_root)
        s3.upload_dir(g_client, s3.BUCKET, g_video_root, VIDEO_ROOT_PREFIX)

        # everything past the first upload has to stay inside this try: unittest skips
        # tearDownModule once setUpModule raised, so the call below is the only thing that
        # removes what was uploaded
        import webdataset_base as base

        g_index = base.generate_temp_index_file(g_tar)
        g_client.upload_file(g_index.name, s3.BUCKET, f"{WDS_PREFIX}/shard0.index")
    except Exception:
        tearDownModule()
        raise


def tearDownModule():
    global g_server, g_tmpdir, g_index, g_client
    try:
        # The mock server, and everything uploaded to it, is thrown away wholesale.
        g_client = None
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


@pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=None)
def mxnet_pipe(path, index_path, dont_use_mmap=False):
    return tuple(fn.readers.mxnet(path=path, index_path=index_path, dont_use_mmap=dont_use_mmap))


@pipeline_def(batch_size=3, num_threads=num_threads, device_id=None)
def numpy_pipe(file_root, dont_use_mmap=False):
    return fn.readers.numpy(file_root=file_root, name="Reader", dont_use_mmap=dont_use_mmap)


@pipeline_def(batch_size=batch_size, num_threads=num_threads)
def video_pipe(device, filenames=None, file_list=None, file_root=None, random_shuffle=False):
    if filenames is not None:
        source = dict(filenames=filenames, labels=list(range(len(filenames))))
    elif file_list is not None:
        source = dict(file_list=file_list, file_list_format="frames")
    else:
        source = dict(file_root=file_root)
    return tuple(
        fn.experimental.readers.video(
            **source,
            device=device,
            name="Reader",
            sequence_length=4,
            stride=3,
            step=15,
            enable_frame_num=True,
            enable_timestamps=True,
            random_shuffle=random_shuffle,
        )
    )


def _video_pipe(device, **kwargs):
    return video_pipe(device, device_id=0 if device == "gpu" else None, seed=1234, **kwargs)


def _epoch_size(pipe):
    return pipe.reader_meta("Reader")["epoch_size"]


def test_file_reader_file_root():
    compare_pipelines(
        file_pipe(file_root=f"s3://{s3.BUCKET}/{DATA_PREFIX}"),
        file_pipe(file_root=g_root, dont_use_mmap=True),  # mmap cannot map the 0-byte file
        batch_size,
        3,
    )


def test_file_reader_files_arg():
    compare_pipelines(
        file_pipe(files=[f"s3://{s3.BUCKET}/{DATA_PREFIX}/{f}" for f in g_files]),
        file_pipe(files=[os.path.join(g_root, f) for f in g_files], dont_use_mmap=True),
        batch_size,
        2,
    )


@attr("slow")
def test_file_reader_listing_pagination():
    # seeded here rather than in setUpModule: it costs ~7 s and nothing else needs it
    _seed_many_objects(s3.s3_client(g_server.endpoint_url))
    pipe = file_pipe(file_root=f"s3://{s3.BUCKET}/{MANY_PREFIX}")
    pipe.build()
    assert pipe.reader_meta("Reader")["epoch_size"] == 1100, pipe.reader_meta("Reader")


def test_webdataset_index_inferred():
    compare_pipelines(
        wds_pipe(paths=f"s3://{s3.BUCKET}/{WDS_PREFIX}/shard0.tar"),
        wds_pipe(paths=g_tar, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_webdataset_local_index():
    compare_pipelines(
        wds_pipe(paths=f"s3://{s3.BUCKET}/{WDS_PREFIX}/shard0.tar", index_paths=[g_index.name]),
        wds_pipe(paths=g_tar, index_paths=[g_index.name], dont_use_mmap=True),
        batch_size,
        2,
    )


def test_webdataset_remote_index():
    compare_pipelines(
        wds_pipe(
            paths=f"s3://{s3.BUCKET}/{WDS_PREFIX}/shard0.tar",
            index_paths=[f"s3://{s3.BUCKET}/{WDS_PREFIX}/shard0.index"],
        ),
        wds_pipe(paths=g_tar, index_paths=[g_index.name], dont_use_mmap=True),
        batch_size,
        2,
    )


@attr("slow")
def test_webdataset_remote_index_multi_chunk():
    # 3000 samples puts the index well past FileStreamBuf's 64 KiB read chunk
    # (webdataset_loader.cc), so building this pipeline exercises multiple Read() round trips
    # against S3 - including the one at the exact end of the index - not just the single-chunk
    # case every other webdataset test here happens to hit.
    import webdataset_base as base

    with tempfile.TemporaryDirectory() as tmpdir:
        big_tar = os.path.join(tmpdir, "big.tar")
        _make_local_tar(big_tar, num_samples=3000)
        big_index = base.generate_temp_index_file(big_tar)
        try:
            prefix = f"{WDS_PREFIX}/big"
            g_client.upload_file(big_tar, s3.BUCKET, f"{prefix}/shard0.tar")
            g_client.upload_file(big_index.name, s3.BUCKET, f"{prefix}/shard0.index")
            compare_pipelines(
                wds_pipe(
                    paths=f"s3://{s3.BUCKET}/{prefix}/shard0.tar",
                    index_paths=[f"s3://{s3.BUCKET}/{prefix}/shard0.index"],
                ),
                wds_pipe(paths=big_tar, index_paths=[big_index.name], dont_use_mmap=True),
                batch_size,
                2,
            )
        finally:
            big_index.close()


def test_mxnet_reader_local_index():
    compare_pipelines(
        mxnet_pipe(path=f"s3://{s3.BUCKET}/{REC_PREFIX}/train.rec", index_path=g_rec_index),
        mxnet_pipe(path=g_rec, index_path=g_rec_index, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_mxnet_reader_remote_index():
    compare_pipelines(
        mxnet_pipe(
            path=f"s3://{s3.BUCKET}/{REC_PREFIX}/train.rec",
            index_path=f"s3://{s3.BUCKET}/{REC_PREFIX}/train.idx",
        ),
        mxnet_pipe(path=g_rec, index_path=g_rec_index, dont_use_mmap=True),
        batch_size,
        2,
    )


def test_file_reader_missing_object():
    with assert_raises(RuntimeError, glob="*S3 Object not found. bucket=*object=*"):
        pipe = file_pipe(files=[f"s3://{s3.BUCKET}/{DATA_PREFIX}/no-such-object.dat"])
        pipe.build()
        pipe.run()


def test_file_reader_missing_bucket():
    with assert_raises(RuntimeError, glob="*NoSuchBucket*"):
        file_pipe(file_root=f"s3://{s3.MISSING_BUCKET}/data").build()


# ---------------------------------------------------------------------------------------------
# Listing: each of these pins a branch of s3_discover_files / s3_filesystem::list_objects_f
# ---------------------------------------------------------------------------------------------


def test_directory_markers_are_not_samples():
    """A zero-byte object named "<prefix>/class_a/" is a folder, not a sample.

    lexically_relative() turns it into ("class_a", ""), so it only stays out of the dataset
    because of the empty-filename guard in s3_discover_files.
    """
    pipe = file_pipe(file_root=f"s3://{s3.BUCKET}/{QUIRKS_PREFIX}")
    pipe.build()
    assert _epoch_size(pipe) == len(g_quirks_files), pipe.reader_meta("Reader")


def test_listing_depth_filtering():
    """Only objects exactly one subdirectory below the prefix become samples.

    The fixture also holds an object directly under the prefix and one two levels down; both are
    rejected by the path_elems != 2 check, which is also what keeps the prefix's own directory
    marker (lexically_relative() maps it to ".") from being dereferenced.
    """
    pipe = file_pipe(file_root=f"s3://{s3.BUCKET}/{QUIRKS_PREFIX}")
    pipe.build()
    data, _ = pipe.run()  # fn.readers.file yields (contents, label)
    read = sorted(bytes(to_array(data[i])).decode() for i in range(len(g_quirks_files)))
    assert read == [f"Contents of {name}.\n" for name in g_quirks_files], read


def test_numpy_reader_nested_prefix():
    """readers.numpy over s3://, one subdirectory below the prefix."""
    compare_pipelines(
        numpy_pipe(file_root=f"s3://{s3.BUCKET}/{NUMPY_PREFIX}/nested"),
        numpy_pipe(file_root=os.path.join(g_numpy_root, "nested"), dont_use_mmap=True),
        3,
        2,
    )


def test_numpy_reader_flat_prefix():
    """readers.numpy over s3://, objects directly under the prefix.

    Without label_from_subdir there is no subdirectory to take a label from, so these count as
    samples - the local backend reads them and discovery over s3:// has to agree. Dropping them
    surfaces as "No files found." rather than a wrong result, which is why this compares against
    the local reader instead of asserting a count.
    """
    compare_pipelines(
        numpy_pipe(file_root=f"s3://{s3.BUCKET}/{NUMPY_PREFIX}/flat"),
        numpy_pipe(file_root=os.path.join(g_numpy_root, "flat"), dont_use_mmap=True),
        3,
        2,
    )


@params("", "/")
def test_file_reader_bucket_root(suffix):
    """file_root naming a whole bucket lists it with an empty prefix, not with "/"."""
    s3_pipe = file_pipe(file_root=f"s3://{ROOT_BUCKET}{suffix}")
    local_pipe = file_pipe(file_root=g_bucket_root)
    s3_pipe.build()
    local_pipe.build()
    assert _epoch_size(s3_pipe) == _epoch_size(local_pipe) == 3, s3_pipe.reader_meta("Reader")
    compare_pipelines(s3_pipe, local_pipe, batch_size, 2)


# ---------------------------------------------------------------------------------------------
# Video reader against s3:// URLs: libavformat opens only local files, so these videos are
# streamed through DALI's FileStream instead
# ---------------------------------------------------------------------------------------------


def _compare_video_epoch(s3_pipe, local_pipe):
    s3_pipe.build()
    local_pipe.build()
    epoch_size = _epoch_size(local_pipe)
    assert _epoch_size(s3_pipe) == epoch_size, (_epoch_size(s3_pipe), epoch_size)
    compare_pipelines(s3_pipe, local_pipe, batch_size, -(-epoch_size // batch_size))


@params("cpu", "gpu")
def test_video_reader_filenames(device):
    """Shuffled, so that the decoder keeps switching files and seeking back and forth."""
    files = VIDEO_FILES[device]
    _compare_video_epoch(
        _video_pipe(device, filenames=[_remote_video(f) for f in files], random_shuffle=True),
        _video_pipe(device, filenames=[_local_video(f) for f in files], random_shuffle=True),
    )


@params("cpu", "gpu")
def test_video_reader_file_list(device):
    """Frame ranges that start mid-video, so that the first read is preceded by a seek."""

    def write_list(path, make_path):
        with open(path, "w") as f:
            for label, name in enumerate(VIDEO_FILES[device]):
                f.write(f"{make_path(name)} {label} {7 + label} 50\n")
        return path

    s3_list = write_list(os.path.join(g_tmpdir.name, f"s3_{device}.txt"), _remote_video)
    local_list = write_list(os.path.join(g_tmpdir.name, f"local_{device}.txt"), _local_video)
    _compare_video_epoch(
        _video_pipe(device, file_list=s3_list), _video_pipe(device, file_list=local_list)
    )


@params("cpu", "gpu")
def test_video_reader_file_root(device):
    """The labels come from the subdirectories, so they compare the listing, not just videos."""
    _compare_video_epoch(
        _video_pipe(
            device,
            file_root=f"s3://{s3.BUCKET}/{VIDEO_ROOT_PREFIX}/{device}",
            random_shuffle=True,
        ),
        _video_pipe(device, file_root=os.path.join(g_video_root, device), random_shuffle=True),
    )


def test_video_reader_source_info():
    files = [_remote_video(f) for f in VIDEO_FILES["cpu"]]
    pipe = _video_pipe("cpu", filenames=files)
    pipe.build()
    video = pipe.run()[0]
    for i in range(batch_size):
        assert video[i].source_info() in files, video[i].source_info()


def test_video_reader_missing_object():
    with assert_raises(RuntimeError, glob="*S3 Object not found. bucket=*object=*"):
        pipe = _video_pipe("cpu", filenames=[_remote_video("no-such-video.mp4")])
        pipe.build()
        pipe.run()
