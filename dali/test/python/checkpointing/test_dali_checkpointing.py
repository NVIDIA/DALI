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

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
from nvidia.dali.pipeline import pipeline_def
from test_utils import get_dali_extra_path, compare_pipelines
from nose2.tools import params, cartesian_params
from nose.plugins.attrib import attr
from dataclasses import dataclass

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

warmup_epochs = 2
comparsion_iterations = 5
pipeline_args = {
    'batch_size': 10,
    'num_threads': 4,
    'enable_checkpointing': True,
    'device_id': 0,
    'exec_async': True,
    'exec_pipelined': True,
}


# Checkpoints can be only accessed between the epochs
# Because of that, we need to calculate the exact epoch size
def calculate_iterations_in_epoch(pipe, batch_size, num_shards=1):
    reader_meta = pipe.reader_meta()
    try:
        epoch_size = reader_meta['Reader']['epoch_size_padded']
        epoch_size = epoch_size // num_shards
    except KeyError:
        # There is no reader in the pipeline
        epoch_size = 1

    # Round up, because pad_last_batch=True
    return (epoch_size + batch_size - 1) // batch_size


def check_pipeline_checkpointing_native(pipeline_factory):
    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iterations_in_epoch = calculate_iterations_in_epoch(pipe, pipeline_args['batch_size'])
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**pipeline_args, checkpoint=pipe.checkpoint())
    compare_pipelines(pipe, restored, pipeline_args['batch_size'], comparsion_iterations)


def check_pipeline_checkpointing_pytorch(pipeline_factory, reader_name=None, size=-1):
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iter = DALIGenericIterator(pipe, ['data'], auto_reset=True,
                               reader_name=reader_name, size=size)
    for _ in range(warmup_epochs):
        for _ in iter:
            pass

    restored = pipeline_factory(**pipeline_args, checkpoint=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIGenericIterator(restored, ['data'], auto_reset=True,
                                reader_name=reader_name, size=size)

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


def check_single_input_operator_pipeline(op, device, **kwargs):

    @pipeline_def
    def pipeline():
        data, _ = fn.readers.file(
            name="Reader", file_root=images_dir,
            pad_last_batch=True, random_shuffle=True)
        decoding_device = 'mixed' if device == 'gpu' else 'cpu'
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
    check_pipeline_checkpointing_pytorch(
        pipeline_factory, reader_name='Reader')


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
        (2, 10, 0, 2, True, False, False, True, 5),
        (4, 256, 2, 4, False, False, True, True, 6),
        (3, 64, 3, 4, False, False, True, False),
        (5, 10, 0, 2, True, False, False, False),
        (1, 3, 0, 1, True, False, False, False),
        (10, 3, 0, 1, True, False, False, False, 1),
        (10, 10, 0, 2, True, False, False, False, 2),
        (10, 256, 2, 4, False, False, True, False, 3),
        (10, 10, 1, 2, False, False, False, False),
        (10, 10, 1, 2, False, False, False, False, 2),
        (7, 10, 0, 2, True, False, True, True, 3, 3),
        (7, 10, 2, 5, True, False, False, False, 3, 10),
        (0, 32, 3, 4, True, False, False, False, 0, 3),
)
def test_file_reader(
        num_epochs, batch_size, shard_id, num_shards,
        random_shuffle, shuffle_after_epoch, stick_to_shard, pad_last_batch,
        iters_into_epoch=None, initial_fill=1024):

    @pipeline_def(batch_size=batch_size, device_id=0,
                  num_threads=4, enable_checkpointing=True)
    def pipeline():
        data, label = fn.readers.file(
            name="Reader", file_root=images_dir,
            pad_last_batch=pad_last_batch, random_shuffle=random_shuffle,
            shard_id=shard_id, num_shards=num_shards,
            shuffle_after_epoch=shuffle_after_epoch,
            stick_to_shard=stick_to_shard,
            initial_fill=initial_fill)

        return data, label

    p = pipeline()
    p.build()

    iterations_in_epoch = calculate_iterations_in_epoch(p, batch_size, num_shards)
    for epoch in range(num_epochs):
        for i in range(iterations_in_epoch):
            p.run()
            if iters_into_epoch is not None:
                if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                    break

    restored = pipeline(checkpoint=p.checkpoint())
    restored.build()

    compare_pipelines(p, restored, batch_size, (num_shards + 1) * iterations_in_epoch)


@attr('pytorch')
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
        num_epochs, batch_size, shard_id, num_shards,
        random_shuffle, shuffle_after_epoch, stick_to_shard, iters_into_epoch=None):

    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    @pipeline_def(batch_size=batch_size, device_id=0,
                  num_threads=4, enable_checkpointing=True)
    def pipeline():
        data, label = fn.readers.file(
            name="Reader", file_root=images_dir,
            pad_last_batch=True, random_shuffle=random_shuffle,
            shard_id=shard_id, num_shards=num_shards,
            shuffle_after_epoch=shuffle_after_epoch,
            stick_to_shard=stick_to_shard)
        image = fn.decoders.image_random_crop(data, device="mixed")
        image = fn.resize(image, size=(200, 200))
        return image, label

    p = pipeline()
    p.build()

    iter = DALIGenericIterator(p, ['data', 'labels'], auto_reset=True,
                               reader_name="Reader")
    for epoch in range(num_epochs):
        for i, _ in enumerate(iter):
            if iters_into_epoch is not None:
                if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                    break

    restored = pipeline(checkpoint=iter.checkpoints()[0])
    restored.build()
    iter2 = DALIGenericIterator(restored, ['data', 'labels'], auto_reset=True,
                                reader_name="Reader")

    for out1, out2 in zip(iter, iter2):
        for d1, d2 in zip(out1, out2):
            for key in d1.keys():
                assert (d1[key] == d2[key]).all()


@params(0, 1, 2, 3, 4, 5, 6, 7, 8)
def test_multiple_readers(num_iters):
    my_images = os.path.join(images_dir, '134')
    files = [os.path.join(my_images, f) for f in os.listdir(my_images)]

    @pipeline_def(batch_size=1, device_id=0,
                  num_threads=4, enable_checkpointing=True)
    def pipeline():
        # Reader with epoch size = 2
        a_enc, _ = fn.readers.file(
            name="Reader1", files=files[:2],
            pad_last_batch=True, random_shuffle=True)

        # Reader with epoch size = 3
        b_enc, _ = fn.readers.file(
            name="Reader2", files=files[:3],
            pad_last_batch=True, random_shuffle=True)

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
        BaseDecoderConfig(shard_id=0, num_shards=1, stick_to_shard=True, pad_last_batch=True,
                          random_shuffle=True),
        BaseDecoderConfig(shard_id=4, num_shards=7, stick_to_shard=True, pad_last_batch=True,
                          random_shuffle=False),
        BaseDecoderConfig(shard_id=6, num_shards=7, stick_to_shard=False, pad_last_batch=False,
                          random_shuffle=False),
        BaseDecoderConfig(shard_id=0, num_shards=2, stick_to_shard=False, pad_last_batch=False,
                          random_shuffle=True),
    ),
    (
        VideoConfig(sequence_length=3, stride=1, step=-1),
        VideoConfig(sequence_length=3, stride=1, step=5),
    ),
)
def test_video_reader(num_epochs, batch_size, iters_into_epoch,
                      config: BaseDecoderConfig, video: VideoConfig):

    files = [os.path.join(get_dali_extra_path(), f'db/video/multiple_framerate/{f}/{f}fps.mp4')
             for f in (10, 50)]

    @pipeline_def(batch_size=batch_size, device_id=0,
                  num_threads=4, enable_checkpointing=True)
    def pipeline():
        images, labels, f, t = fn.readers.video(
            device='gpu',
            filenames=files,
            labels=list(range(len(files))),
            normalized=True,
            random_shuffle=config.random_shuffle,
            image_type=types.RGB,
            dtype=types.FLOAT,
            name="Reader",
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
            step=video.step)

        return images, labels, f, t

    p = pipeline()
    p.build()

    assert p.reader_meta()['Reader']['epoch_size'] // config.num_shards > 2, \
           "Trivial test case: at least 2 samples per shard required"

    iterations_in_epoch = calculate_iterations_in_epoch(p, batch_size, config.num_shards)

    assert iterations_in_epoch >= iters_into_epoch, "Not enough iterations in epoch"

    for epoch in range(num_epochs):
        for i in range(iterations_in_epoch):
            p.run()
            if iters_into_epoch is not None:
                if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                    break

    restored = pipeline(checkpoint=p.checkpoint())
    restored.build()

    compare_pipelines(p, restored, batch_size, (config.num_shards + 1) * iterations_in_epoch)


# Randomized operators section
# note: fn.decoders.image_random_crop is tested by
# `check_single_input_operator`

@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_coin_flip(device, shape):
    check_no_input_operator(fn.random.coin_flip, device, shape=shape)


@attr('pytorch')
@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_coin_flip_pytorch(device, shape):
    check_no_input_operator_pytorch(fn.random.coin_flip, device, shape=shape)


@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_normal(device, shape):
    check_no_input_operator(fn.random.normal, device, shape=shape)


@attr('pytorch')
@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_normal_pytorch(device, shape):
    check_no_input_operator_pytorch(fn.random.normal, device, shape=shape)


@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_uniform(device, shape):
    check_no_input_operator(fn.random.uniform, device, shape=shape)


@attr('pytorch')
@cartesian_params(('cpu',), (None, (1,), (10,)))
def test_random_uniform_pytorch(device, shape):
    check_no_input_operator(fn.random.uniform, device, shape=shape)


# Stateless operators section


@params('cpu', 'gpu')
def test_rotate_checkpointing(device):
    check_single_input_operator(fn.rotate, device, angle=15)


@params('cpu', 'gpu')
def test_resize_checkpointing(device):
    check_single_input_operator(fn.resize, device, resize_x=20, resize_y=10)


@params('cpu', 'gpu')
def test_flip_checkpointing(device):
    check_single_input_operator(fn.flip, device)


@params('cpu', 'gpu')
def test_crop_mirror_normalize_checkpointing(device):
    check_single_input_operator(fn.crop_mirror_normalize, device)


@params('cpu', 'gpu')
def test_warp_affine_checkpointing(device):
    check_single_input_operator(fn.warp_affine, device, matrix=(0.3, 0.7, 5, 0.7, 0.3, -5))


@params('cpu', 'gpu')
def test_saturation_checkpointing(device):
    check_single_input_operator(fn.saturation, device)


@params('cpu', 'gpu')
def test_reductions_min_checkpointing(device):
    check_single_input_operator(fn.reductions.min, device)


@params('cpu', 'gpu')
def test_reductions_max_checkpointing(device):
    check_single_input_operator(fn.reductions.max, device)


@params('cpu', 'gpu')
def test_reductions_sum_checkpointing(device):
    check_single_input_operator(fn.reductions.sum, device, dtype=types.DALIDataType.UINT8)


@params('cpu', 'gpu')
def test_equalize_checkpointing(device):
    check_single_input_operator(fn.experimental.equalize, device)


def test_transforms_crop_checkpointing():
    check_no_input_operator(fn.transforms.crop, 'cpu')


def test_transforms_rotation_checkpointing():
    check_no_input_operator(fn.transforms.rotation, 'cpu', angle=90)


def test_transforms_shear_checkpointing():
    check_no_input_operator(fn.transforms.shear, 'cpu', shear=(2, 2))


def test_transforms_scale_checkpointing():
    check_no_input_operator(fn.transforms.scale, 'cpu', scale=(2, 4))


def test_transforms_translation_checkpointing():
    check_no_input_operator(fn.transforms.translation, 'cpu', offset=(21, 30))
