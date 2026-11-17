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

import importlib.util
import io
import os
from pathlib import Path
from shutil import which
import tarfile
import tempfile
import unittest

from nose_utils import assert_raises


def _load_wds2idx():
    script_path = Path(__file__).parents[4] / "tools" / "wds2idx.py"
    spec = importlib.util.spec_from_file_location("wds2idx", script_path)
    wds2idx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wds2idx)
    return wds2idx


def test_wds2idx_gnu_tar_path_accepts_non_ascii_tar_member_name_with_spaces():
    if which("tar") is None:
        raise unittest.SkipTest("GNU tar not installed")

    wds2idx = _load_wds2idx()

    with tempfile.TemporaryDirectory() as test_dir:
        tar_path = os.path.join(test_dir, "data.tar")
        idx_path = os.path.join(test_dir, "data.idx")
        member_name = "my imgé file.jpg"
        payload = b"\xff\xd8\xff\xe0payload"

        with tarfile.open(tar_path, "w") as archive:
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        with wds2idx.IndexCreator(tar_path, idx_path, verbose=False) as creator:
            entries = list(creator._get_data_tar())

        assert len(entries) == 1
        _, name, size = entries[0]
        assert name == member_name
        assert size == len(payload)


def test_wds2idx_gnu_tar_path_accepts_member_owner_with_space():
    if which("tar") is None:
        raise unittest.SkipTest("GNU tar not installed")

    wds2idx = _load_wds2idx()

    with tempfile.TemporaryDirectory() as test_dir:
        tar_path = os.path.join(test_dir, "data.tar")
        idx_path = os.path.join(test_dir, "data.idx")
        member_name = "sample.bin"
        payload = b"payload"

        with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            info.uname = "owner name"
            info.gname = "group"
            archive.addfile(info, io.BytesIO(payload))

        with wds2idx.IndexCreator(tar_path, idx_path, verbose=False) as creator:
            entries = list(creator._get_data_tar())

        assert len(entries) == 1
        _, name, size = entries[0]
        assert name == member_name
        assert size == len(payload)


def _check_gnu_tar_path_offsets_with_long_member_names(archive_format):
    if which("tar") is None:
        raise unittest.SkipTest("GNU tar not installed")

    wds2idx = _load_wds2idx()

    with tempfile.TemporaryDirectory() as test_dir:
        tar_path = os.path.join(test_dir, "data.tar")
        idx_path = os.path.join(test_dir, "data.idx")
        # names longer than 100 characters don't fit in the tar header and are stored in
        # extra headers placed before the member header (`././@LongLink` in the GNU format)
        payloads = {
            "sample/" + "a" * 150 + ".jpg": b"\xff\xd8\xff\xe0first payload",
            "short.cls": b"1",
            "b" * 600 + ".jpg": b"\xff\xd8\xff\xe0second payload",
        }

        with tarfile.open(tar_path, "w", format=archive_format) as archive:
            for member_name, payload in payloads.items():
                info = tarfile.TarInfo(name=member_name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

        with wds2idx.IndexCreator(tar_path, idx_path, verbose=False) as creator:
            entries = list(creator._get_data_tar())

        assert [name for _, name, _ in entries] == list(payloads)
        with open(tar_path, "rb") as archive:
            for offset, name, size in entries:
                archive.seek(offset)
                assert archive.read(size) == payloads[name]


def test_wds2idx_gnu_tar_path_offsets_with_gnu_long_member_names():
    _check_gnu_tar_path_offsets_with_long_member_names(tarfile.GNU_FORMAT)


def test_wds2idx_gnu_tar_path_offsets_with_pax_long_member_names():
    _check_gnu_tar_path_offsets_with_long_member_names(tarfile.PAX_FORMAT)


def _check_gnu_long_name_header_is_rejected(size, glob):
    wds2idx = _load_wds2idx()

    with tempfile.TemporaryDirectory() as test_dir:
        tar_path = os.path.join(test_dir, "data.tar")
        idx_path = os.path.join(test_dir, "data.idx")
        # a hand-crafted `././@LongLink` header followed by the two empty end-of-archive blocks
        long_name_header = tarfile.TarInfo(name="././@LongLink")
        long_name_header.type = tarfile.GNUTYPE_LONGNAME
        long_name_header.size = size
        with open(tar_path, "wb") as archive:
            archive.write(long_name_header.tobuf(tarfile.GNU_FORMAT))
            archive.write(bytes(2 * tarfile.BLOCKSIZE))

        with wds2idx.IndexCreator(tar_path, idx_path, verbose=False) as creator:
            with open(tar_path, "rb") as archive:
                with assert_raises(ValueError, glob=glob):
                    creator._get_data_offset(archive, 0)


def test_wds2idx_gnu_long_name_header_with_negative_size():
    # the GNU format stores a size in base-256, which can encode a negative number
    _check_gnu_long_name_header_is_rejected(-tarfile.BLOCKSIZE, "Negative size -512 *")


def test_wds2idx_gnu_long_name_header_past_end_of_archive():
    _check_gnu_long_name_header_is_rejected(100 * tarfile.BLOCKSIZE, "Truncated tar header *")
