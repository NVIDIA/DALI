# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import webdataset_base
import numpy as np
from nvidia.dali.pipeline import pipeline_def
from test_utils import get_dali_extra_path, compare_pipelines
from nose_utils import assert_warns
from nose2.tools import params, cartesian_params
from nose.plugins.attrib import attr
from dataclasses import dataclass
from nvidia.dali import tfrecord as tfrec
from reader.test_numpy import is_gds_supported

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
    pipe.build()

    iterations_in_epoch = calculate_iterations_in_epoch(pipe, pipeline_args["batch_size"])
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**pipeline_args, checkpoint=pipe.checkpoint())
    compare_pipelines(pipe, restored, pipeline_args["batch_size"], comparsion_iterations)


def check_pipeline_checkpointing_pytorch(pipeline_factory, reader_name=None, size=-1):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iter = DALIGenericIterator(pipe, ["data"], auto_reset=True, reader_name=reader_name, size=size)
    for _ in range(warmup_epochs):
        for _ in iter:
            pass

    restored = pipeline_factory(**pipeline_args, checkpoint=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIGenericIterator(
        restored, ["data"], auto_reset=True, reader_name=reader_name, size=size
    )

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


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


def check_single_input_operator_pytorch(op, device, **kwargs):
    pipeline_factory = check_single_input_operator_pipeline(op, device, **kwargs)
    check_pipeline_checkpointing_pytorch(pipeline_factory, reader_name="Reader")


def check_no_input_operator(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        return op(device=device, **kwargs)

    check_pipeline_checkpointing_native(pipeline_factory)


def check_no_input_operator_pytorch(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        return op(device=device, **kwargs)

    check_pipeline_checkpointing_pytorch(pipeline_factory, size=8)


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
    p.build()

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
    restored.build()

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


@attr("pytorch")
@params(
    (1, 3, 0, 1, True, False, False),
    (5, 10, 0, 2, True, False, False),
    (3, 64, 3, 4, False, False, False),
    (0, 32, 1, 4, False, False, True),
    (3, 64, 3, 4, False, False, True),
    (1, 8, 0, 2, False, True, False),
    (1, 8, 1, 2, False, True, False),
    (1, 8, 3, 4, False, True, False),
    (1, 3, 0, 1, True, False, False, 1),
    (5, 10, 0, 2, True, False, False, 2),
    (3, 64, 3, 4, False, False, True, 3),
)
def test_file_reader_pytorch(
    num_epochs,
    batch_size,
    shard_id,
    num_shards,
    random_shuffle,
    shuffle_after_epoch,
    stick_to_shard,
    iters_into_epoch=None,
):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_checkpointing=True)
    def pipeline():
        data, label = fn.readers.file(
            name="Reader",
            file_root=images_dir,
            pad_last_batch=True,
            random_shuffle=random_shuffle,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=shuffle_after_epoch,
            stick_to_shard=stick_to_shard,
        )
        image = fn.decoders.image_random_crop(data, device="mixed")
        image = fn.resize(image, size=(200, 200))
        return image, label

    p = pipeline()
    p.build()

    iter = DALIGenericIterator(p, ["data", "labels"], auto_reset=True, reader_name="Reader")
    for epoch in range(num_epochs):
        for i, _ in enumerate(iter):
            if iters_into_epoch is not None:
                if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                    break

    restored = pipeline(checkpoint=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIGenericIterator(restored, ["data", "labels"], auto_reset=True, reader_name="Reader")

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


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
    p.build()

    for _ in range(num_iters):
        p.run()

    restored = pipeline(checkpoint=p.checkpoint())
    restored.build()

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
def test_video_reader(
    num_epochs, batch_size, iters_into_epoch, config: BaseDecoderConfig, video: VideoConfig
):
    files = [
        os.path.join(get_dali_extra_path(), f"db/video/multiple_framerate/{f}/{f}fps.mp4")
        for f in (10, 50)
    ]

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


@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
def test_random_coin_flip(device, shape):
    check_no_input_operator(fn.random.coin_flip, device, shape=shape)


@attr("pytorch")
@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
def test_random_coin_flip_pytorch(device, shape):
    check_no_input_operator_pytorch(fn.random.coin_flip, device, shape=shape)


@cartesian_params(("cpu",), (None, (1,), (10,)))
def test_random_normal(device, shape):
    check_no_input_operator(fn.random.normal, device, shape=shape)


@attr("pytorch")
@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
def test_random_normal_pytorch(device, shape):
    check_no_input_operator_pytorch(fn.random.normal, device, shape=shape)


@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
def test_random_uniform(device, shape):
    check_no_input_operator(fn.random.uniform, device, shape=shape)


@attr("pytorch")
@cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
def test_random_uniform_pytorch(device, shape):
    check_no_input_operator_pytorch(fn.random.uniform, device, shape=shape)


def test_random_object_bbox():
    check_single_input_operator(fn.segmentation.random_object_bbox, "cpu", format="box")


def test_random_mask_pixel():
    check_single_input_operator(fn.segmentation.random_mask_pixel, "cpu")


def test_roi_random_crop():
    check_single_input_operator(
        fn.roi_random_crop, "cpu", crop_shape=(10, 10), roi_start=(0, 0), roi_end=(30, 30)
    )


def test_ssd_random_crop():
    @pipeline_def
    def pipeline():
        data = fn.random.uniform(shape=(100, 100), dtype=types.DALIDataType.UINT8)
        bbox = fn.random.uniform(shape=(4,), range=[0, 100], dtype=types.DALIDataType.FLOAT)
        labels = fn.random.uniform(shape=(1,), dtype=types.DALIDataType.INT32)
        return fn.ssd_random_crop(data, bbox, labels, device="cpu")[0]

    check_pipeline_checkpointing_native(pipeline)


def test_batch_permutation():
    check_no_input_operator(fn.batch_permutation, "cpu")


def test_jitter():
    check_single_input_operator(fn.jitter, "gpu")


def test_random_bbox_crop():
    def wrapper(input, **kwargs):
        bboxes = fn.cast(input[:, :4, 0], dtype=types.DALIDataType.FLOAT)
        bboxes /= fn.reductions.max(bboxes, axes=(0, 1))
        out = fn.random_bbox_crop(bboxes, bbox_layout="xyXY", input_shape=(2000, 2000), **kwargs)
        return out[0]

    check_single_input_operator(wrapper, "cpu")


# Stateless operators section


@params("cpu", "gpu")
def test_rotate_checkpointing(device):
    check_single_input_operator(fn.rotate, device, angle=15)


@params("cpu", "gpu")
def test_resize_checkpointing(device):
    check_single_input_operator(fn.resize, device, resize_x=20, resize_y=10)


@params("cpu", "gpu")
def test_flip_checkpointing(device):
    check_single_input_operator(fn.flip, device)


@params("cpu", "gpu")
def test_crop_mirror_normalize_checkpointing(device):
    check_single_input_operator(fn.crop_mirror_normalize, device)


@params("cpu", "gpu")
def test_warp_affine_checkpointing(device):
    check_single_input_operator(fn.warp_affine, device, matrix=(0.3, 0.7, 5, 0.7, 0.3, -5))


@params("cpu", "gpu")
def test_saturation_checkpointing(device):
    check_single_input_operator(fn.saturation, device)


@params("cpu", "gpu")
def test_reductions_min_checkpointing(device):
    check_single_input_operator(fn.reductions.min, device)


@params("cpu", "gpu")
def test_reductions_max_checkpointing(device):
    check_single_input_operator(fn.reductions.max, device)


@params("cpu", "gpu")
def test_reductions_sum_checkpointing(device):
    check_single_input_operator(fn.reductions.sum, device, dtype=types.DALIDataType.UINT8)


@params("cpu", "gpu")
def test_equalize_checkpointing(device):
    check_single_input_operator(fn.experimental.equalize, device)


def test_transforms_crop_checkpointing():
    check_no_input_operator(fn.transforms.crop, "cpu")


def test_transforms_rotation_checkpointing():
    check_no_input_operator(fn.transforms.rotation, "cpu", angle=90)


def test_transforms_shear_checkpointing():
    check_no_input_operator(fn.transforms.shear, "cpu", shear=(2, 2))


def test_transforms_scale_checkpointing():
    check_no_input_operator(fn.transforms.scale, "cpu", scale=(2, 4))


def test_transforms_translation_checkpointing():
    check_no_input_operator(fn.transforms.translation, "cpu", offset=(21, 30))


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
    p1.build()

    for _ in range(iterations):
        run_and_reset(p1)

    cpt = p1.checkpoint()
    p2 = pipeline_factory(checkpoint=cpt)
    p2.build()
    compare_external_source_pipelines(p1, p2, compare_iterations)


def check_external_source_pipeline_checkpointing_pytorch(pipeline_factory, iterations, *, size=-1):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    def run(iterator, iterations):
        completed_iterations = 0
        while completed_iterations < iterations:
            for _ in iterator:
                completed_iterations += 1
                if completed_iterations == iterations:
                    break

    pipeline = pipeline_factory()
    pipeline.build()

    iter = DALIGenericIterator(pipeline, ["data"], auto_reset=True, size=size)

    run(iter, iterations)

    restored = pipeline_factory(checkpoint=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIGenericIterator(restored, ["data"], auto_reset=True, size=size)

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


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


@cartesian_params(
    ((1, 1), (3, 4)),  # (epoch size, batch size)
    (0, 3, 15),  # test iterations
    ("idx", "batch_info", "sample_info"),  # indexing mode
    (True, False),  # parallel
)
def test_external_source_checkpointing(dataset_info, iterations, mode, parallel):
    epoch_size, batch_size = dataset_info
    source = make_dummy_source(epoch_size, batch_size, mode)
    pf = make_external_source_test_pipeline_factory(source, mode, batch_size, parallel)
    check_external_source_pipeline_checkpointing(pf, iterations, 2 * epoch_size)


@attr("pytorch")
@cartesian_params(
    ((1, 1), (4, 5)),  # (epoch size, batch size)
    (0, 4, 11),  # test iterations
    ("idx", "batch_info", "sample_info"),  # indexing mode
    (True, False),  # parallel
)
def test_external_source_checkpointing_pytorch(dataset_info, iterations, mode, parallel):
    epoch_size, batch_size = dataset_info
    source = make_dummy_source(epoch_size, batch_size, mode)
    pf = make_external_source_test_pipeline_factory(source, mode, batch_size, parallel)
    check_external_source_pipeline_checkpointing_pytorch(pf, iterations)


@cartesian_params(
    ("iterator", "iterable", "callable"),  # source kind
    (True, False),  # parallel
)
def test_external_source_unsupported(kind, parallel):
    if kind == "iterator":
        source = iter([1, 2, 3])
    elif kind == "iterable":
        source = [1, 2, 3]
    elif kind == "callable":

        def source():
            return 42

    @pipeline_def(batch_size=1, num_threads=1, device_id=0, enable_checkpointing=True)
    def pipeline():
        return fn.external_source(source=source)

    with assert_warns(glob="DALI doesn't capture state of such 'source'."):
        pipeline().build()
