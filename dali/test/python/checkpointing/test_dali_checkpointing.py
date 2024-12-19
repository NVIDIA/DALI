# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tempfile
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import re
import shutil
import webdataset_base
import numpy as np
from nvidia.dali.pipeline import pipeline_def
from test_utils import (
    compare_pipelines,
    create_sign_off_decorator,
    get_dali_extra_path,
    module_functions,
)
from nose_utils import assert_warns, assert_raises, attr
from nose2.tools import params, cartesian_params
from dataclasses import dataclass
from nvidia.dali import tfrecord as tfrec
from nvidia.dali.auto_aug import auto_augment as aa
from nvidia.dali.auto_aug import rand_augment as ra
from nvidia.dali.auto_aug import trivial_augment as ta
from reader.test_numpy import is_gds_supported


reader_signed_off = create_sign_off_decorator()
random_signed_off = create_sign_off_decorator()

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")

warmup_epochs = 2
comparsion_iterations = 5
pipeline_args = {
    "batch_size": 10,
    "num_threads": 4,
    "enable_checkpointing": True,
    "device_id": 0,
    "exec_async": True,
    "exec_pipelined": True,
}


# Checkpoints can be only accessed between the epochs
# Because of that, we need to calculate the exact epoch size
def calculate_iterations_in_epoch(pipe, batch_size, num_shards=1):
    reader_meta = pipe.reader_meta()
    try:
        epoch_size = reader_meta["Reader"]["epoch_size_padded"]
        epoch_size = epoch_size // num_shards
    except KeyError:
        # There is no reader in the pipeline
        epoch_size = 1

    # Round up, because pad_last_batch=True
    return (epoch_size + batch_size - 1) // batch_size


def check_pipeline_checkpointing_native(pipeline_factory):
    pipe = pipeline_factory(**pipeline_args)

    iterations_in_epoch = calculate_iterations_in_epoch(pipe, pipeline_args["batch_size"])
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**pipeline_args, checkpoint=pipe.checkpoint())
    compare_pipelines(pipe, restored, pipeline_args["batch_size"], comparsion_iterations)


def check_single_input_operator_pipeline(op, device, **kwargs):
    @pipeline_def
    def pipeline():
        data, _ = fn.readers.file(
            name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True
        )
        decoding_device = "mixed" if device == "gpu" else "cpu"
        decoded = fn.decoders.image_random_crop(data, device=decoding_device)
        casted = fn.cast(decoded, dtype=types.DALIDataType.UINT8)
        resized = fn.resize(casted, resize_x=120, resize_y=80)
        return op(resized, device=device, **kwargs)

    return pipeline


def check_single_input_operator(op, device, **kwargs):
    pipeline_factory = check_single_input_operator_pipeline(op, device, **kwargs)
    check_pipeline_checkpointing_native(pipeline_factory)


def check_no_input_operator(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        return op(device=device, **kwargs)

    check_pipeline_checkpointing_native(pipeline_factory)


# Readers section


def check_reader_checkpointing(reader, num_epochs, batch_size, iters_into_epoch, **kwargs):
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_checkpointing=True)
    def pipeline():
        result = reader(name="Reader", **kwargs)
        if isinstance(result, list):
            return tuple(result)
        else:
            return result

    p = pipeline()

    num_shards = kwargs.get("num_shards", 1)

    assert (
        p.reader_meta()["Reader"]["epoch_size"] // num_shards > 2
    ), "Trivial test case: at least 2 samples per shard required"

    iterations_in_epoch = calculate_iterations_in_epoch(p, batch_size, num_shards)
    assert iterations_in_epoch >= (iters_into_epoch or 0), "Not enough iterations in epoch"

    for epoch in range(num_epochs):
        for i in range(iterations_in_epoch):
            p.run()
            if iters_into_epoch is not None:
                if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                    break

    restored = pipeline(checkpoint=p.checkpoint())

    compare_pipelines(p, restored, batch_size, (num_shards + 1) * iterations_in_epoch)


@params(
    (1, 3, 0, 1, True, False, False, True),
    (5, 10, 0, 2, True, False, False, True),
    (0, 32, 1, 4, False, False, False, True),
    (3, 64, 3, 4, False, False, False, True),
    (1, 3, 0, 1, True, False, True, True),
    (5, 10, 0, 2, True, False, True, True),
    (0, 32, 1, 4, False, False, True, True),
    (3, 64, 3, 4, False, False, True, True),
    (2, 7, 0, 1, False, True, False, True),
    (1, 8, 0, 2, False, True, False, True),
    (1, 8, 1, 2, False, True, False, True),
    (1, 8, 3, 4, False, True, False, True),
    (2, 11, 2, 5, False, True, False, True),
    (5, 3, 0, 1, True, False, False, True, 4),
    (2, 4, 0, 2, True, False, False, True, 5),
    (4, 5, 2, 4, False, False, True, True, 3),
    (3, 64, 3, 4, False, False, True, False),
    (5, 10, 0, 2, True, False, False, False),
    (1, 3, 0, 1, True, False, False, False),
    (10, 3, 0, 1, True, False, False, False, 1),
    (10, 10, 0, 2, True, False, False, False, 2),
    (10, 4, 2, 4, False, False, True, False, 3),
    (10, 10, 1, 2, False, False, False, False),
    (10, 10, 1, 2, False, False, False, False, 2),
    (7, 10, 0, 2, True, False, True, True, 3, 3),
    (7, 4, 2, 5, True, False, False, False, 3, 2),
    (0, 32, 3, 4, True, False, False, False, 0, 3),
)
@reader_signed_off("readers.file", "file_reader")
def test_file_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    shuffle_after_epoch,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    check_reader_checkpointing(
        fn.readers.file,
        num_epochs,
        batch_size,
        iters_into_epoch,
        file_root=images_dir,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle_after_epoch=shuffle_after_epoch,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )


# Coco reader is based on file reader and all the strange corner cases are (hopefully) tested there
@params(
    (0, 4, 1, 2, True, False, False, False, None),
    (4, 5, 0, 1, False, True, False, False, 1),
    (16, 6, 3, 5, False, False, True, False, 2),
    (6, 7, 2, 3, False, True, False, True, 3),
)
@reader_signed_off("readers.coco", "coco_reader")
def test_coco_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    shuffle_after_epoch,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    coco_dir = os.path.join(data_root, "db", "coco")
    coco_images = os.path.join(coco_dir, "images")
    coco_annotations = os.path.join(coco_dir, "instances.json")

    check_reader_checkpointing(
        fn.readers.coco,
        num_epochs,
        batch_size,
        iters_into_epoch,
        file_root=coco_images,
        annotations_file=coco_annotations,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle_after_epoch=shuffle_after_epoch,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
        polygon_masks=True,
        image_ids=True,
    )


@params(
    (0, 1, 0, 1, True, True, True, None),
    (0, 3, 0, 2, True, True, False, 1),
    (4, 5, 1, 3, True, False, True, 2),
    (1, 7, 2, 4, True, False, False, None),
    (11, 2, 2, 10, False, True, True, 1),
    (2, 4, 1, 6, False, True, False, 2),
    (5, 6, 2, 3, False, False, True, None),
    (3, 8, 4, 5, False, False, False, 1),
)
@reader_signed_off("readers.mxnet", "mxnet_reader")
def test_mxnet_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
):
    recordio_dir = os.path.join(data_root, "db", "recordio")
    recordio_rec = os.path.join(recordio_dir, "train.rec")
    recordio_idx = os.path.join(recordio_dir, "train.idx")

    check_reader_checkpointing(
        fn.readers.mxnet,
        num_epochs,
        batch_size,
        iters_into_epoch,
        path=recordio_rec,
        index_path=recordio_idx,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
    )


@params(
    (0, 1, 0, 1, True, True, True, None),
    (0, 2, 0, 2, True, True, False, 1),
    (6, 3, 1, 3, True, False, True, 2),
    (3, 4, 2, 4, True, False, False, None),
    (10, 5, 2, 10, False, True, True, 1),
    (4, 6, 1, 6, False, True, False, 2),
    (10, 7, 2, 3, False, False, True, None),
    (2, 8, 4, 5, False, False, False, 1),
)
@reader_signed_off("readers.tfrecord", "tfrecord_reader")
def test_tfrecord_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
):
    tfrecord_dir = os.path.join(data_root, "db", "tfrecord")
    tfrecord = os.path.join(tfrecord_dir, "train")
    tfrecord_idx = os.path.join(tfrecord_dir, "train.idx")

    def tfrecord_wrapper(*args, **kwargs):
        return fn.readers.tfrecord(*args, **kwargs)["image/encoded"]

    check_reader_checkpointing(
        tfrecord_wrapper,
        num_epochs,
        batch_size,
        iters_into_epoch,
        path=tfrecord,
        index_path=tfrecord_idx,
        features={
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
        },
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
    )


@params(
    (1, 1, 0, 3, False, False, False, None),
    (2, 2, 0, 1, False, False, True, 1),
    (6, 4, 2, 3, False, True, False, 2),
    (4, 1, 1, 5, False, True, True, 3),
    (3, 2, 4, 5, True, False, False, 1),
    (7, 4, 2, 5, True, False, True, 2),
    (5, 1, 2, 3, True, True, False, 3),
    (0, 2, 3, 6, True, True, True, None),
)
@reader_signed_off("readers.sequence", "sequence_reader")
def test_sequence_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    check_reader_checkpointing(
        fn.readers.sequence,
        num_epochs,
        batch_size,
        iters_into_epoch,
        file_root=os.path.join(data_root, "db", "sequence", "frames"),
        sequence_length=5,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )


@params(
    (1, 3, 0, 1, True, True, True, 1),
    (5, 5, 1, 3, True, True, False, 2),
    (6, 7, 2, 3, True, False, True, 3),
    (5, 3, 0, 1, True, False, False, 1),
    (7, 5, 2, 3, False, True, True, None),
    (4, 1, 1, 2, False, True, False, 2),
    (0, 3, 3, 4, False, False, True, None),
    (1, 4, 2, 3, False, False, False, 3),
)
@reader_signed_off("readers.caffe", "caffe_reader")
def test_caffe_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    caffe_dir = os.path.join(data_root, "db", "lmdb")

    check_reader_checkpointing(
        fn.readers.caffe,
        num_epochs,
        batch_size,
        iters_into_epoch,
        path=caffe_dir,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )


@params(
    (1, 2, 0, 2, True, True, True, 1),
    (4, 4, 1, 2, True, True, False, 2),
    (5, 6, 0, 2, True, False, True, None),
    (6, 2, 1, 3, True, False, False, 1),
    (3, 4, 3, 4, False, True, True, 2),
    (8, 1, 2, 3, False, True, False, None),
    (0, 2, 4, 5, False, False, True, None),
    (3, 3, 1, 3, False, False, False, 2),
)
@reader_signed_off("readers.caffe2", "caffe2_reader")
def test_caffe2_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    caffe2_dir = os.path.join(data_root, "db", "c2lmdb")

    check_reader_checkpointing(
        fn.readers.caffe2,
        num_epochs,
        batch_size,
        iters_into_epoch,
        path=caffe2_dir,
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )


@params(
    (10, 1, 1, 8, False, False, False, 3),
    (3, 2, 0, 1, False, False, True, 4),
    (3, 4, 4, 7, False, True, False, 5),
    (0, 8, 2, 6, False, True, True, None),
    (12, 16, 0, 5, True, False, False, 3),
    (8, 32, 1, 3, True, False, True, 4),
    (6, 64, 4, 6, True, True, False, 5),
    (10, 128, 3, 4, True, True, True, None),
)
@reader_signed_off("readers.webdataset")
def test_webdataset_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    tar_file_paths = [
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-1.tar"),
        os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-2.tar"),
    ]
    index_files = [
        webdataset_base.generate_temp_index_file(tar_file_path) for tar_file_path in tar_file_paths
    ]

    check_reader_checkpointing(
        fn.readers.webdataset,
        num_epochs,
        batch_size,
        iters_into_epoch,
        paths=tar_file_paths,
        index_paths=[f.name for f in index_files],
        ext=["jpg", "cls"],
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )


@params(
    (0, 1, 0, 1, False, False, False, False, None),
    (5, 2, 1, 2, False, False, False, True, 1),
    (6, 3, 4, 5, False, False, True, False, 2),
    (7, 4, 2, 5, False, False, True, True, 3),
    (5, 1, 1, 3, False, True, False, False, 4),
    (6, 2, 2, 4, False, True, False, True, None),
    (7, 3, 0, 1, True, False, False, False, 1),
    (8, 4, 2, 3, True, False, False, True, 2),
    (9, 1, 0, 1, True, False, True, False, 3),
    (10, 2, 0, 2, True, False, True, True, 4),
)
@reader_signed_off("readers.nemo_asr", "nemo_asr_reader")
def test_nemo_asr_reader(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    shuffle_after_epoch,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    nemo_dir = os.path.join(data_root, "db", "audio", "wav")
    wav_files = [os.path.join(nemo_dir, f) for f in os.listdir(nemo_dir) if f.endswith(".wav")]

    manifest = tempfile.NamedTemporaryFile("w")
    for i, f in enumerate(wav_files):
        manifest.write(
            f'{{"audio_filepath": "{f}", \
                       "offset": {i/1000}, \
                       "duration": {0.3 + i/100}, \
                       "text": "o{"o"*i}"}}\n'
        )
    manifest.flush()

    check_reader_checkpointing(
        fn.readers.nemo_asr,
        num_epochs,
        batch_size,
        iters_into_epoch,
        manifest_filepaths=[manifest.name],
        pad_last_batch=pad_last_batch,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle_after_epoch=shuffle_after_epoch,
        stick_to_shard=stick_to_shard,
        initial_fill=initial_fill,
    )

    manifest.close()


# device,
# num_epochs, batch_size, shard_id, num_shards,
# random_shuffle, shuffle_after_epoch, stick_to_shard, pad_last_batch,
# iters_into_epoch, initial_fill
@params(
    ("cpu", 0, 1, 0, 1, False, False, False, False, None),
    ("cpu", 5, 2, 4, 7, False, False, False, True, 1),
    ("cpu", 4, 4, 0, 2, False, False, True, False, 2),
    ("cpu", 3, 8, 4, 6, False, False, True, True, 3),
    ("cpu", 6, 1, 2, 3, False, True, False, False, 4),
    ("cpu", 5, 2, 2, 5, False, True, False, True, 3),
    ("cpu", 4, 4, 3, 4, True, False, False, False, 2),
    ("cpu", 3, 8, 1, 4, True, False, False, True, 1),
    ("cpu", 2, 1, 1, 2, True, False, True, False, None),
    ("cpu", 0, 2, 0, 1, True, False, True, True, 2),
    *(
        [
            ("gpu", 2, 1, 1, 2, False, False, False, False, None),
            ("gpu", 5, 2, 0, 5, False, False, False, True, 1),
            ("gpu", 3, 4, 2, 3, False, False, True, False, 2),
            ("gpu", 6, 8, 3, 5, False, False, True, True, 3),
            ("gpu", 7, 1, 1, 4, False, True, False, False, 4),
            ("gpu", 3, 2, 2, 4, False, True, False, True, 3),
            ("gpu", 3, 4, 2, 5, True, False, False, False, 2),
            ("gpu", 4, 8, 0, 2, True, False, False, True, 1),
            ("gpu", 1, 1, 2, 3, True, False, True, False, None),
            ("gpu", 0, 2, 0, 2, True, False, True, True, 2),
        ]
        if is_gds_supported()
        else []
    ),
)
@reader_signed_off("readers.numpy", "numpy_reader")
def test_numpy_reader(
    device,
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    shuffle_after_epoch,
    stick_to_shard,
    pad_last_batch,
    iters_into_epoch=None,
    initial_fill=1024,
):
    numpy_dir = os.path.join(data_root, "db", "3D", "MRI", "Knee", "npy_2d_slices", "STU00001")

    # GDS doesn't support overlayfs, so we need to use runner's scratch
    gds_data_root = "/scratch/"
    if not os.path.isdir(gds_data_root):
        gds_data_root = os.getcwd() + "/scratch/"
        if not os.path.isdir(gds_data_root):
            os.mkdir(gds_data_root)
            assert os.path.isdir(gds_data_root)

    with tempfile.TemporaryDirectory(prefix=gds_data_root) as test_data_root:
        shutil.copytree(numpy_dir, os.path.join(test_data_root, "numpy"))

        check_reader_checkpointing(
            fn.readers.numpy,
            num_epochs,
            batch_size,
            iters_into_epoch,
            device=device,
            file_root=os.path.join(test_data_root, "numpy"),
            pad_last_batch=pad_last_batch,
            random_shuffle=random_shuffle,
            shuffle_after_epoch=shuffle_after_epoch,
            shard_id=shard_id,
            num_shards=num_shards,
            stick_to_shard=stick_to_shard,
            initial_fill=initial_fill,
        )


@params(0, 1, 2, 3, 4, 5, 6, 7, 8)
def test_multiple_readers(num_iters):
    my_images = os.path.join(images_dir, "134")
    files = [os.path.join(my_images, f) for f in os.listdir(my_images)]

    @pipeline_def(batch_size=1, device_id=0, num_threads=4, enable_checkpointing=True)
    def pipeline():
        # Reader with epoch size = 2
        a_enc, _ = fn.readers.file(
            name="Reader1", files=files[:2], pad_last_batch=True, random_shuffle=True
        )

        # Reader with epoch size = 3
        b_enc, _ = fn.readers.file(
            name="Reader2", files=files[:3], pad_last_batch=True, random_shuffle=True
        )

        a = fn.decoders.image_random_crop(a_enc)
        b = fn.decoders.image_random_crop(b_enc)
        a = fn.resize(a, size=(200, 200))
        b = fn.resize(b, size=(200, 200))
        return (a + b) // 2

    p = pipeline()

    for _ in range(num_iters):
        p.run()

    restored = pipeline(checkpoint=p.checkpoint())

    compare_pipelines(p, restored, 1, 20)


@dataclass
class BaseDecoderConfig:
    shard_id: int
    num_shards: int
    stick_to_shard: bool
    pad_last_batch: bool
    random_shuffle: bool


@dataclass
class VideoConfig:
    sequence_length: int
    stride: int
    step: int


@cartesian_params(
    (0, 1, 3),
    (1, 3),
    (0, 2),
    (
        BaseDecoderConfig(
            shard_id=0, num_shards=1, stick_to_shard=True, pad_last_batch=True, random_shuffle=True
        ),
        BaseDecoderConfig(
            shard_id=4, num_shards=7, stick_to_shard=True, pad_last_batch=True, random_shuffle=False
        ),
        BaseDecoderConfig(
            shard_id=6,
            num_shards=7,
            stick_to_shard=False,
            pad_last_batch=False,
            random_shuffle=False,
        ),
        BaseDecoderConfig(
            shard_id=0,
            num_shards=2,
            stick_to_shard=False,
            pad_last_batch=False,
            random_shuffle=True,
        ),
    ),
    (
        VideoConfig(sequence_length=3, stride=1, step=-1),
        VideoConfig(sequence_length=3, stride=1, step=5),
    ),
)
@reader_signed_off("readers.video", "video_reader")
def test_video_reader(
    num_epochs, batch_size, iters_into_epoch, config: BaseDecoderConfig, video: VideoConfig
):
    files = [os.path.join(get_dali_extra_path(), f"db/video/small/small{i}.mp4") for i in range(5)]

    check_reader_checkpointing(
        fn.readers.video,
        num_epochs,
        batch_size,
        iters_into_epoch,
        device="gpu",
        filenames=files,
        labels=list(range(len(files))),
        normalized=True,
        random_shuffle=config.random_shuffle,
        image_type=types.RGB,
        dtype=types.FLOAT,
        enable_frame_num=True,
        enable_timestamps=True,
        file_list_frame_num=True,
        file_list_include_preceding_frame=False,
        num_shards=config.num_shards,
        shard_id=config.shard_id,
        stick_to_shard=config.stick_to_shard,
        pad_last_batch=config.pad_last_batch,
        sequence_length=video.sequence_length,
        stride=video.stride,
        step=video.step,
    )


# simplified case of test_video_reader suite
@cartesian_params(
    (2,),
    (1, 3),
    (0, 3),
    (
        BaseDecoderConfig(
            shard_id=0, num_shards=1, stick_to_shard=True, pad_last_batch=True, random_shuffle=True
        ),
        BaseDecoderConfig(
            shard_id=6,
            num_shards=7,
            stick_to_shard=False,
            pad_last_batch=False,
            random_shuffle=False,
        ),
        BaseDecoderConfig(
            shard_id=0,
            num_shards=2,
            stick_to_shard=False,
            pad_last_batch=False,
            random_shuffle=True,
        ),
    ),
    (VideoConfig(sequence_length=3, stride=1, step=-1),),
)
@reader_signed_off("readers.video_resize", "video_reader_resize")
def test_video_reader_resize_reader(
    num_epochs, batch_size, iters_into_epoch, config: BaseDecoderConfig, video: VideoConfig
):
    files = [os.path.join(get_dali_extra_path(), f"db/video/small/small{i}.mp4") for i in range(5)]

    check_reader_checkpointing(
        fn.readers.video_resize,
        num_epochs,
        batch_size,
        iters_into_epoch,
        device="gpu",
        filenames=files,
        labels=list(range(len(files))),
        normalized=True,
        random_shuffle=config.random_shuffle,
        image_type=types.RGB,
        dtype=types.FLOAT,
        enable_frame_num=True,
        enable_timestamps=True,
        file_list_frame_num=True,
        file_list_include_preceding_frame=False,
        num_shards=config.num_shards,
        shard_id=config.shard_id,
        stick_to_shard=config.stick_to_shard,
        pad_last_batch=config.pad_last_batch,
        sequence_length=video.sequence_length,
        stride=video.stride,
        step=video.step,
        size=(100, 100),
    )


@cartesian_params(
    (
        "cpu",
        "gpu",
    ),
    (0, 4),
    (4,),
    (0, 2),
    (
        BaseDecoderConfig(
            shard_id=1, num_shards=2, stick_to_shard=True, pad_last_batch=True, random_shuffle=True
        ),
        BaseDecoderConfig(
            shard_id=2,
            num_shards=3,
            stick_to_shard=False,
            pad_last_batch=False,
            random_shuffle=False,
        ),
    ),
    (VideoConfig(sequence_length=3, stride=1, step=5),),
)
@reader_signed_off("experimental.readers.video")
def test_experimental_video_reader(
    device, num_epochs, batch_size, iters_into_epoch, config: BaseDecoderConfig, video: VideoConfig
):
    files = [
        os.path.join(get_dali_extra_path(), "db", "video", "vfr", f"test_{i}.mp4") for i in (1, 2)
    ]

    check_reader_checkpointing(
        fn.experimental.readers.video,
        num_epochs,
        batch_size,
        iters_into_epoch,
        device=device,
        filenames=files,
        labels=list(range(len(files))),
        random_shuffle=config.random_shuffle,
        num_shards=config.num_shards,
        shard_id=config.shard_id,
        stick_to_shard=config.stick_to_shard,
        pad_last_batch=config.pad_last_batch,
        sequence_length=video.sequence_length,
        stride=video.stride,
        step=video.step,
    )


# Randomized operators section
# note: fn.decoders.image_random_crop is tested by
# `check_single_input_operator`


@params(*[lambda n: np.arange(n) / (n * (n - 1) / 2), lambda n: None])
@random_signed_off("random.choice")
def test_random_choice(p_dist):
    @pipeline_def
    def pipeline():
        n = 10
        return fn.random.choice(n, p=p_dist(n))

    check_pipeline_checkpointing_native(pipeline)


@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
@random_signed_off("random.coin_flip", "coin_flip")
def test_random_coin_flip(device, shape):
    check_no_input_operator(fn.random.coin_flip, device, shape=shape)


@cartesian_params(("cpu",), (None, (1,), (10,)))
@random_signed_off("random.normal", "normal_distribution")
def test_random_normal(device, shape):
    check_no_input_operator(fn.random.normal, device, shape=shape)


@cartesian_params(("cpu",), (None, 100, (10, 50)))
@random_signed_off("random.beta")
def test_random_beta(device, shape):
    check_no_input_operator(fn.random.beta, device, shape=shape)


@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
@random_signed_off("random.uniform", "uniform")
def test_random_uniform(device, shape):
    check_no_input_operator(fn.random.uniform, device, shape=shape)


@random_signed_off("segmentation.random_object_bbox")
def test_random_object_bbox():
    check_single_input_operator(fn.segmentation.random_object_bbox, "cpu", format="box")


@random_signed_off("segmentation.random_mask_pixel")
def test_random_mask_pixel():
    check_single_input_operator(fn.segmentation.random_mask_pixel, "cpu")


@random_signed_off("roi_random_crop")
def test_roi_random_crop():
    check_single_input_operator(
        fn.roi_random_crop, "cpu", crop_shape=(10, 10), roi_start=(0, 0), roi_end=(30, 30)
    )


@random_signed_off("ssd_random_crop")
def test_ssd_random_crop():
    @pipeline_def
    def pipeline():
        data = fn.random.uniform(shape=(100, 100), dtype=types.DALIDataType.UINT8)
        bbox = fn.random.uniform(shape=(7, 4), range=[0, 100], dtype=types.DALIDataType.FLOAT)
        labels = fn.random.uniform(shape=(1,), dtype=types.DALIDataType.INT32)
        return fn.ssd_random_crop(data, bbox, labels, device="cpu")[0]

    check_pipeline_checkpointing_native(pipeline)


@random_signed_off("batch_permutation")
def test_batch_permutation():
    check_no_input_operator(fn.batch_permutation, "cpu")


@random_signed_off("jitter")
def test_jitter():
    check_single_input_operator(fn.jitter, "gpu")


@random_signed_off("random_resized_crop")
@params("cpu", "gpu")
def test_random_resized_crop(device):
    check_single_input_operator(fn.random_resized_crop, device, size=(42, 24))


@random_signed_off("random_bbox_crop")
def test_random_bbox_crop():
    def wrapper(input, **kwargs):
        bboxes = fn.cast(input[:, :4, 0], dtype=types.DALIDataType.FLOAT)
        bboxes /= fn.reductions.max(bboxes, axes=(0, 1))
        out = fn.random_bbox_crop(bboxes, bbox_layout="xyXY", input_shape=(2000, 2000), **kwargs)
        return out[0]

    check_single_input_operator(wrapper, "cpu")


@random_signed_off("random_crop_generator")
def test_random_crop_generator():
    @pipeline_def
    def pipeline():
        data = fn.random.uniform(shape=(2,), dtype=types.DALIDataType.INT64)
        crop_anchor, crop_shape = fn.random_crop_generator(data)
        return crop_anchor, crop_shape

    check_pipeline_checkpointing_native(pipeline)


@params("cpu", "gpu")
@random_signed_off("noise.gaussian")
def test_noise_gaussian(device):
    check_single_input_operator(fn.noise.gaussian, device, stddev=150)


@params("cpu", "gpu")
@random_signed_off("noise.salt_and_pepper")
def test_noise_salt_and_pepper(device):
    check_single_input_operator(fn.noise.salt_and_pepper, device, prob=0.5)


@params("cpu", "gpu")
@random_signed_off("noise.shot")
def test_noise_shot(device):
    check_single_input_operator(fn.noise.shot, device, factor=100)


@params("cpu", "mixed")
@random_signed_off("image_decoder_random_crop", "decoders.image_random_crop")
def test_image_random_crop(device):
    @pipeline_def
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image_random_crop(data, device=device)
        return image

    check_pipeline_checkpointing_native(pipeline)


@params("cpu", "mixed")
@random_signed_off(
    "experimental.image_decoder_random_crop", "experimental.decoders.image_random_crop"
)
def test_experimental_image_random_crop(device):
    @pipeline_def
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.experimental.decoders.image_random_crop(data, device=device)
        return image

    check_pipeline_checkpointing_native(pipeline)


# External source


def check_external_source_pipeline_checkpointing(pipeline_factory, iterations, compare_iterations):
    def run_and_reset(pipe):
        try:
            return pipe.run()
        except StopIteration:
            pipe.reset()
            return pipe.run()

    def compare_external_source_pipelines(pipe1, pipe2, steps):
        for _ in range(steps):
            out1 = run_and_reset(pipe1)[0].as_array()
            out2 = run_and_reset(pipe2)[0].as_array()
            assert np.all(out1 == out2)

    p1 = pipeline_factory()

    for _ in range(iterations):
        run_and_reset(p1)

    cpt = p1.checkpoint()
    p2 = pipeline_factory(checkpoint=cpt)
    compare_external_source_pipelines(p1, p2, compare_iterations)


def make_external_source_test_pipeline_factory(source, mode, batch_size, parallel, **kwargs):
    kwargs["parallel"] = parallel
    if mode == "idx":
        kwargs["batch"] = True
        kwargs["batch_info"] = False
    elif mode == "batch_info":
        kwargs["batch"] = True
        kwargs["batch_info"] = True
    elif mode == "sample_info":
        kwargs["batch"] = False
        kwargs["batch_info"] = False
    else:
        assert False, "Unknown mode!"

    @pipeline_def(
        batch_size=batch_size,
        num_threads=4,
        device_id=0,
        enable_checkpointing=True,
        py_start_method="spawn",
    )
    def pipeline_factory():
        return fn.external_source(source=source, **kwargs)

    return pipeline_factory


def make_dummy_source(epoch_size, batch_size, mode):
    if mode == "idx":

        def src(idx):
            if idx >= epoch_size:
                raise StopIteration()
            return [np.asarray([idx, i]) for i in range(batch_size)]

    elif mode == "batch_info":

        def src(idx):
            if idx.iteration >= epoch_size:
                raise StopIteration()
            return [np.asarray([idx.epoch_idx, idx.iteration, i]) for i in range(batch_size)]

    elif mode == "sample_info":

        def src(idx):
            if idx.idx_in_epoch >= epoch_size * batch_size:
                raise StopIteration()
            return np.asarray([idx.epoch_idx, idx.iteration, idx.idx_in_epoch, idx.idx_in_batch])

    return src


@attr("sanitizer_skip")
@cartesian_params(
    ((1, 1), (3, 4)),  # (epoch size, batch size)
    (0, 3, 15),  # test iterations
    ("idx", "batch_info", "sample_info"),  # indexing mode
    (True, False),  # parallel
)
@reader_signed_off("external_source")
def test_external_source_checkpointing(dataset_info, iterations, mode, parallel):
    epoch_size, batch_size = dataset_info
    source = make_dummy_source(epoch_size, batch_size, mode)
    pf = make_external_source_test_pipeline_factory(source, mode, batch_size, parallel)
    check_external_source_pipeline_checkpointing(pf, iterations, 2 * epoch_size)


@cartesian_params(
    ("iterator", "iterable", "callable"),  # source kind
    (True, False),  # parallel
)
def test_external_source_unsupported(kind, parallel):
    if kind == "iterator":
        source = iter([np.array(1), np.array(2), np.array(3)])
    elif kind == "iterable":
        source = [np.array(1), np.array(2), np.array(3)]
    elif kind == "callable":

        def source():
            return np.array(42)

    @pipeline_def(batch_size=1, num_threads=1, device_id=0, enable_checkpointing=True)
    def pipeline():
        return fn.external_source(source=source, batch=False)

    with assert_warns(glob="DALI doesn't capture state of such 'source'."):
        pipeline().build()


# Auto augmentation tests - run auto augmentations as a good example of pipeline
# consisting of many ops


@params("cpu", "gpu")
def test_auto_augment(device):
    @pipeline_def(enable_conditionals=True)
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(data, device="cpu" if device == "cpu" else "mixed")
        return aa.auto_augment(image)

    check_pipeline_checkpointing_native(pipeline)


@params("cpu", "gpu")
def test_rand_augment(device):
    @pipeline_def(enable_conditionals=True)
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(data, device="cpu" if device == "cpu" else "mixed")
        return ra.rand_augment(image, n=2, m=15)

    check_pipeline_checkpointing_native(pipeline)


@params("cpu", "gpu")
def test_trivial_augment(device):
    @pipeline_def(enable_conditionals=True)
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        image = fn.decoders.image(data, device="cpu" if device == "cpu" else "mixed")
        return ta.trivial_augment_wide(image)

    check_pipeline_checkpointing_native(pipeline)


@cartesian_params((0, 1), (0, 3), (0, 1), (0, 4))
def test_multiple_restores(warmup_epochs, warmup_iters, run_epochs, run_iters):
    batch_size = 4

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, enable_checkpointing=True)
    def pipeline():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir)
        return fn.decoders.image(data, device="cpu")

    pipe = pipeline()

    iters_in_epoch = pipe.reader_meta()["Reader"]["epoch_size"] // batch_size
    warmup_iters += warmup_epochs * iters_in_epoch
    run_iters += run_epochs * iters_in_epoch

    for _ in range(warmup_iters):
        pipe.run()

    pipe2 = pipeline(checkpoint=pipe.checkpoint())
    for _ in range(run_iters):
        pipe2.run()

    pipe3 = pipeline(checkpoint=pipe2.checkpoint())

    compare_pipelines(pipe2, pipe3, batch_size, 5)


def test_unsupported_dangling_subgraph():
    es = fn.external_source("asdf")

    @pipeline_def(batch_size=1, num_threads=1, device_id=None, enable_checkpointing=True)
    def pipe(arg):
        return arg + 0

    p = pipe(es)

    with assert_raises(
        RuntimeError,
        glob="The pipeline does not support checkpointing*"
        "because it contains operator*outside the pipeline*",
    ):
        p.build()


unsupported_readers = [
    "experimental.readers.fits",
]

unsupported_ops = [
    "experimental.decoders.video",
    "experimental.inputs.video",
    "plugin.video.decoder",
]


def test_coverage():
    from checkpointing.test_dali_stateless_operators import stateless_signed_off

    tested_ops = (
        stateless_signed_off.tested_ops
        | reader_signed_off.tested_ops
        | random_signed_off.tested_ops
    )

    excluded_ops = unsupported_readers + unsupported_ops

    fn_ops = module_functions(
        fn, remove_prefix="nvidia.dali.fn", allowed_private_modules=["_conditional"]
    )
    assert len(fn_ops), "There should be some DALI ops in the `fn`, got nothing"
    if excluded_ops:
        exclude = "|".join(
            "(^" + pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".") + "$)"
            for pattern in excluded_ops
        )
        exclude = re.compile(exclude)
        fn_ops = [x for x in fn_ops if not exclude.match(x)]
    not_covered = sorted(list(set(fn_ops) - tested_ops))
    not_covered_str = ",\n".join(f"'{op_name}'" for op_name in not_covered)
    # we are fine with covering more we can easily list, like numba
    assert (
        set(fn_ops).difference(tested_ops) == set()
    ), f"Test doesn't cover {len(not_covered)} ops:\n{not_covered_str}"
