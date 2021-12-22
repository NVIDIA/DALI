# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.pipeline.experimental import pipeline_def
from test_utils import compare_pipelines, get_dali_extra_path

import numpy as np
import os
from nose_utils import raises
from nose.plugins.attrib import attr

file_root = os.path.join(get_dali_extra_path(), 'db/single/jpeg')


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def rn50_pipeline_base():
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(resized_images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, jpegs, labels, images, resized_images, output


def test_debug_pipeline_base():
    pipe_standard = rn50_pipeline_base()
    pipe_debug = rn50_pipeline_base(debug=True)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def rn50_pipeline():
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    print(f'rng: {rng.get().as_array()}')
    tmp = rng ^ 1
    print(f'rng xor: {tmp.get().as_array()}')
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    if jpegs.get().is_dense_tensor():
        print(f'jpegs: {jpegs.get().as_array()}')
    else:
        print('jpegs shapes:')
        for j in jpegs.get():
            print(j.shape())
    print(f'labels: {labels.get().as_array()}')
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    for i in images.get().as_cpu():
        print(i)
    for i in images.get():
        print(i.shape())
    images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    for i in images.get():
        print(i.shape())
    print(np.array(images.get().as_cpu()[0]))
    images += 1
    print(np.array(images.get().as_cpu()[0]))
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return (output, labels.gpu())


def test_operations_on_debug_pipeline():
    pipe = rn50_pipeline()
    pipe.build()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def load_images_pipeline():
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    return images, labels


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def injection_pipeline(callback, device='cpu'):
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(callback(), device=device, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, images, output


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def injection_pipeline_standard(device='cpu'):
    jpegs, _ = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, device=device, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, images, output


def _test_injection(device, name, transform, eps=1e-07):
    print(f'\nTesting {name}')
    pipe_load = load_images_pipeline()
    pipe_load.build()
    pipe_standard = injection_pipeline_standard(device)
    pipe_debug = injection_pipeline(lambda: transform(pipe_load.run()[0]), device)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10, eps=eps)


def test_injection_numpy():
    _test_injection('cpu', 'numpy array', lambda xs: [np.array(x) for x in xs])


@attr('mxnet')
def test_injection_mxnet():
    import mxnet
    _test_injection('cpu', 'mxnet array', lambda xs: [mxnet.nd.array(x, dtype='uint8') for x in xs])


@attr('pytorch')
def test_injection_torch():
    import torch
    yield _test_injection, 'cpu', 'torch cpu tensor', lambda xs: [torch.tensor(np.array(x), device='cpu') for x in xs]
    yield _test_injection, 'gpu', 'torch gpu tensor', lambda xs: [torch.tensor(np.array(x), device='cuda') for x in xs], 1e-03


@attr('cupy')
def test_injection_cupy():
    import cupy
    _test_injection('gpu', 'cupy array', lambda xs: [cupy.array(x) for x in xs], 1e-03)


def test_injection_dali_types():
    yield _test_injection, 'gpu', 'list of TensorGPU', lambda xs: [x._as_gpu() for x in xs], 1e-03
    yield _test_injection, 'cpu', 'TensorListCPU', lambda xs: xs
    yield _test_injection, 'gpu', 'TensorListGPU', lambda xs: xs._as_gpu(), 1e-03


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def es_pipeline():
    images = fn.external_source(name='input')
    labels = fn.external_source(name='labels')
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, images, output, labels


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def es_pipeline_standard():
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, images, output, labels


def test_external_source_debug():
    n_iters = 10
    pipe_load = load_images_pipeline()
    pipe_standard = es_pipeline_standard()
    pipe_debug = es_pipeline()
    pipe_load.build()
    pipe_debug.build()
    for _ in range(n_iters):
        images, labels = pipe_load.run()
        pipe_debug.feed_input('input', [np.array(t) for t in images])
        pipe_debug.feed_input('labels', np.array(labels.as_tensor()))
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def order_change_pipeline():
    if order_change_pipeline.change:
        rng = 0
    else:
        order_change_pipeline.change = True
        rng = fn.random.coin_flip(probability=0.5, seed=47)
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(resized_images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return rng, jpegs, labels, images, resized_images, output


@raises(RuntimeError, glob='Unexpected operator *. Debug mode does not support'
        ' changing the order of operators executed within the pipeline.')
def test_operators_order_change():
    order_change_pipeline.change = False
    pipe = order_change_pipeline()
    pipe.build()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def inputs_len_change():
    input = [np.zeros(1)] * 8
    if inputs_len_change.change:
        inputs_len_change.change = False
        inputs = [input]
    else:
        inputs = [input]*2
    return fn.cat(*inputs)


@raises(RuntimeError, glob='Trying to use operator * with different number of inputs than when it was built.')
def test_inputs_len_change():
    inputs_len_change.change = True
    pipe = inputs_len_change()
    pipe.build()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def kwargs_len_change():
    input = [np.zeros(1)] * 8
    inputs = [input]*2
    kwargs = {}
    if kwargs_len_change.change:
        kwargs_len_change.change = False
        kwargs['axis'] = 0
    print(len(kwargs))
    return fn.cat(*inputs, **kwargs)


@raises(RuntimeError, glob='Trying to use operator * with different number of keyward arguments than when it was built.')
def test_kwargs_len_change():
    kwargs_len_change.change = True
    pipe = kwargs_len_change()
    pipe.build()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def inputs_batch_change():
    if inputs_batch_change.change:
        inputs_batch_change.change = False
        input = np.zeros(8)
    else:
        input = [np.zeros(1)]*8
    return fn.random.coin_flip(input)


@raises(RuntimeError, glob='In operator * input *')
def test_inputs_batch_change():
    inputs_batch_change.change = True
    pipe = inputs_batch_change()
    pipe.build()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def kwargs_batch_change():
    kwargs = {}
    if kwargs_batch_change.change:
        kwargs_batch_change.change = False
        kwargs['probability'] = 0.75
    else:
        kwargs['probability'] = [np.zeros(1)]*8
    return fn.random.coin_flip(**kwargs)


@raises(RuntimeError, glob='In operator * argument *')
def test_kwargs_batch_change():
    kwargs_batch_change.change = True
    pipe = kwargs_batch_change()
    pipe.build()
    pipe.run()
    pipe.run()


@pipeline_def
def init_config_pipeline():
  jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
  return jpegs, labels


def test_init_config_pipeline():
    pipe_standard = init_config_pipeline(batch_size=8, num_threads=3, device_id=0)
    pipe_debug = init_config_pipeline(batch_size=8, num_threads=3, device_id=0, debug=True)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)
