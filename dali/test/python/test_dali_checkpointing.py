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
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from test_utils import get_dali_extra_path, compare_pipelines
from nose2.tools import params

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
def calculate_iterations_in_epoch(pipe):
    reader_meta = pipe.reader_meta()
    try:
        epoch_size = reader_meta['Reader']['epoch_size_padded']
    except KeyError:
        # There is no reader in the pipeline
        epoch_size = 1

    # Round up, because pad_last_batch=True
    return (epoch_size + pipeline_args['batch_size'] - 1) // pipeline_args['batch_size']


def check_pipeline_checkpointing_native(pipeline_factory):
    pipe = pipeline_factory(**pipeline_args)
    pipe.build()

    iterations_in_epoch = calculate_iterations_in_epoch(pipe)
    for _ in range(warmup_epochs * iterations_in_epoch):
        pipe.run()

    restored = pipeline_factory(**pipeline_args, checkpoint=pipe.checkpoint())
    compare_pipelines(pipe, restored, pipeline_args['batch_size'], comparsion_iterations)


def check_pipeline_checkpointing_pytorch(pipeline_factory, reader_name=None, size=-1):
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


def check_single_input_operator(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data, _ = fn.readers.file(name="Reader", file_root=images_dir, pad_last_batch=True, random_shuffle=True)
        decoding_device = 'mixed' if device == 'gpu' else 'cpu'
        decoded = fn.decoders.image_random_crop(data, device=decoding_device)
        casted = fn.cast(decoded, dtype=types.DALIDataType.UINT8)
        resized = fn.resize(casted, resize_x=120, resize_y=80)
        return op(resized, device=device, **kwargs)

    check_pipeline_checkpointing_native(pipeline_factory)
    check_pipeline_checkpointing_pytorch(pipeline_factory, reader_name='Reader')


def check_no_input_operator(op, device, **kwargs):
    @pipeline_def
    def pipeline_factory():
        return op(device=device, **kwargs)

    check_pipeline_checkpointing_native(pipeline_factory)
    check_pipeline_checkpointing_pytorch(pipeline_factory, size=8)


# Readers section
# note: fn.readers.file is tested by `check_single_input_operator`


# Randomized operators section
# note: fn.decoders.image_random_crop is tested by `check_single_input_operator`

@params('cpu')
def test_random_coin_flip(device):
    check_no_input_operator(fn.random.coin_flip, device, shape=[10])


@params('cpu')
def test_random_normal(device):
    check_no_input_operator(fn.random.normal, device, shape=[10])


@params('cpu')
def test_random_normal(device):
    check_no_input_operator(fn.random.uniform, device, shape=[10])


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
