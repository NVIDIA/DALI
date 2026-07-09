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


def _load_wds2idx():
    script_path = Path(__file__).parents[4] / "tools" / "wds2idx.py"
    spec = importlib.util.spec_from_file_location("wds2idx", script_path)
    wds2idx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wds2idx)
    return wds2idx


def test_wds2idx_gnu_tar_path_accepts_utf8_tar_member_name():
    if which("tar") is None:
        raise unittest.SkipTest("GNU tar not installed")

    wds2idx = _load_wds2idx()

    with tempfile.TemporaryDirectory() as test_dir:
        tar_path = os.path.join(test_dir, "data.tar")
        idx_path = os.path.join(test_dir, "data.idx")
        member_name = "imgé.jpg"
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
