# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
import tempfile

import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import cartesian_params, params
from nose_utils import SkipTest
from test_utils import get_dali_extra_path

dali_extra_path = get_dali_extra_path()


@params("cpu", "gpu")
def test_reader_batch(device_type):
    reader = ndd.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
    )

    iters = 0
    for file, lbl in reader.next_epoch(batch_size=4):
        assert isinstance(file, ndd.Batch)
        assert isinstance(lbl, ndd.Batch)
        assert file.batch_size == 4
        assert lbl.batch_size == 4
        assert file.dtype == ndd.uint8
        assert lbl.dtype == ndd.int32
        assert file.device == ndd.Device("cpu")
        assert lbl.device == ndd.Device("cpu")
        file.evaluate()
        img = ndd.decoders.image(file, device=device_type)
        img.evaluate()
        assert img.dtype == ndd.uint8
        assert len(img.shape[0]) == 3  # HWC
        assert img.shape[0][2] == 3  # RGB
        assert img.device == ndd.Device("cpu" if device_type == "cpu" else "gpu")
        iters += 1
    assert iters > 0


@params("cpu", "gpu")
def test_reader_sample(device_type):
    reader = ndd.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
    )

    iters = 0
    for file, lbl in reader.next_epoch(batch_size=None):
        assert isinstance(file, ndd.Tensor)
        assert isinstance(lbl, ndd.Tensor)
        assert file.dtype == ndd.uint8
        assert lbl.dtype == ndd.int32
        assert file.device == ndd.Device("cpu")
        assert lbl.device == ndd.Device("cpu")
        file.evaluate()
        img = ndd.decoders.image(file, device=device_type)
        img.evaluate()
        assert img.dtype == ndd.uint8
        assert img.shape[2] == 3  # RGB
        assert img.device == ndd.Device("cpu" if device_type == "cpu" else "gpu")
        iters += 1
    assert iters > 0


@cartesian_params(("cpu", "gpu"), (None, 4))
def test_tfrecord(device_type, batch_size):
    try:
        import nvidia.dali.tfrecord as tfrec
    except (RuntimeError, ImportError):
        raise SkipTest("DALI was not compiled with TFRecord support.") from None

    tfrecord = os.path.join(dali_extra_path, "db", "tfrecord", "train")

    with tempfile.NamedTemporaryFile(suffix=".idx") as idx_file:
        subprocess.call(["tfrecord2idx", tfrecord, idx_file.name])

        reader = ndd.readers.TFRecord(
            path=tfrecord,
            index_path=idx_file.name,
            features={
                "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
            },
        )

        expected_type = ndd.Tensor if batch_size is None else ndd.Batch

        for features in reader.next_epoch(batch_size=batch_size):
            assert isinstance(features, dict)

            jpeg = features["image/encoded"]
            assert isinstance(jpeg, expected_type)
            assert jpeg.dtype == ndd.uint8
            assert jpeg.device == ndd.Device("cpu")

            label = features["image/class/label"]
            assert isinstance(label, expected_type)
            assert label.dtype == ndd.int64
            assert label.device == ndd.Device(name="cpu")
            if batch_size is None:
                assert label.shape == (1,)
            else:
                assert label.shape == [(1,)] * batch_size

            img = ndd.decoders.image(jpeg, device=device_type).evaluate()
            assert isinstance(img, expected_type)
            assert img.dtype == ndd.uint8
            assert img.device == ndd.Device(device_type)
            if batch_size is None:
                assert img.shape[-1] == 3
            else:
                assert all(shape[-1] == 3 for shape in img.shape)
