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
import math

import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import cartesian_params, params
from nose_utils import SkipTest, assert_raises
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


@cartesian_params(
    (True, False),
    (True, False),
)
def test_reader_shards(stick_to_shard, pad_last_batch):
    SHARD_NUM = 8
    reader = ndd.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
    )
    all_samples = tuple(reader.next_epoch())
    for initial_shard_id in range(SHARD_NUM):
        reader = ndd.readers.File(
            file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
            file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
            stick_to_shard=stick_to_shard,
            pad_last_batch=pad_last_batch,
            shard_id=initial_shard_id,
            num_shards=SHARD_NUM,
        )
        shard_id = initial_shard_id
        for _ in range(SHARD_NUM):
            samples = tuple(reader.next_epoch())
            samples_in_shard = len(samples)
            samples_num = reader._op_backend.GetReaderMeta()["epoch_size_padded"]
            unpadded_samples_num = reader._op_backend.GetReaderMeta()["epoch_size"]
            shards_beg = math.floor(shard_id * samples_num / SHARD_NUM)
            shards_end = math.floor((shard_id + 1) * samples_num / SHARD_NUM)
            unpadded_shards_beg = math.floor(shard_id * unpadded_samples_num / SHARD_NUM)
            unpadded_shards_end = math.floor((shard_id + 1) * unpadded_samples_num / SHARD_NUM)
            shard_size = shards_end - shards_beg
            for idx in range(0, shard_size):
                if pad_last_batch and shards_beg + idx >= unpadded_shards_end:
                    # duplicate last sample in the shard in case of padding
                    ref_sample = all_samples[unpadded_shards_end - 1]
                else:
                    ref_sample = all_samples[unpadded_shards_beg + idx]
                assert ref_sample == samples[idx], (
                    f"Sample {shards_beg + idx} mismatch, shard_id: {shard_id} (initial_shard_id: "
                    + f"{initial_shard_id}, shard_beg: {shards_beg}, shard_end: {shards_end})"
                )
            assert samples_in_shard == shards_end - shards_beg, (
                f"Samples in shard {shard_id} (initial_shard_id: {initial_shard_id}) should be "
                + f"{shards_end - shards_beg}, but got {samples_in_shard}"
            )
            if not stick_to_shard:
                shard_id = (shard_id + 1) % SHARD_NUM


def test_reader_shards_error():
    reader = ndd.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
        num_shards=99999,
    )
    with assert_raises(
        RuntimeError, glob='Assert on "num_shards_ <= Size()" failed: The number of input samples:'
    ):
        tuple(reader.next_epoch())
