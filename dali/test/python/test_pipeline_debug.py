# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import os
from nvidia.dali import fn
from nvidia.dali import tensors
from nvidia.dali import types
from nvidia.dali.pipeline.experimental import pipeline_def
from nose_utils import attr, raises, assert_raises
from test_utils import compare_pipelines, get_dali_extra_path

from conditionals.test_pipeline_conditionals import (
    pred_gens,
    _impl_against_split_merge,
    _impl_dot_gpu,
    _impl_arg_inputs_scoped_tracking,
    _impl_arg_inputs_scoped_uninitialized,
    _impl_generators,
    _impl_uninitialized,
)


file_root = os.path.join(get_dali_extra_path(), "db/single/jpeg")


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def rn50_pipeline_base():
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        resized_images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, jpegs, labels, images, resized_images, output


def test_debug_pipeline_base():
    pipe_standard = rn50_pipeline_base()
    pipe_debug = rn50_pipeline_base(debug=True)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def rn50_pipeline():
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    print(f"rng: {rng.get().as_array()}")
    tmp = rng ^ 1
    print(f"rng xor: {tmp.get().as_array()}")
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    if jpegs.get().is_dense_tensor():
        print(f"jpegs: {jpegs.get().as_array()}")
    else:
        print("jpegs shapes:")
        for j in jpegs.get():
            print(j.shape())
    print(f"labels: {labels.get().as_array()}")
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
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

    output = fn.crop_mirror_normalize(
        images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return (output, labels.gpu())


def test_operations_on_debug_pipeline():
    pipe = rn50_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def load_images_pipeline():
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    return images, labels


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def injection_pipeline(callback, device="cpu"):
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(callback(), device=device, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, images, output


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def injection_pipeline_standard(device="cpu"):
    jpegs, _ = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    if device == "gpu":
        images = images.gpu()
    images = fn.random_resized_crop(images, device=device, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, images, output


def _test_injection(device, name, transform, eps=1e-07):
    pipe_load = load_images_pipeline()
    pipe_standard = injection_pipeline_standard(device)
    pipe_debug = injection_pipeline(lambda: transform(pipe_load.run()[0]), device)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10, eps=eps)


def test_injection_numpy():
    _test_injection("cpu", "numpy array", lambda xs: [np.array(x) for x in xs])


@attr("pytorch")
def test_injection_torch():
    import torch

    yield _test_injection, "cpu", "torch cpu tensor", lambda xs: [
        torch.tensor(np.array(x), device="cpu") for x in xs
    ]
    yield _test_injection, "gpu", "torch gpu tensor", lambda xs: [
        torch.tensor(np.array(x), device="cuda") for x in xs
    ]


@attr("cupy")
def test_injection_cupy():
    import cupy

    _test_injection("gpu", "cupy array", lambda xs: [cupy.array(x) for x in xs])


def test_injection_dali_types():
    yield _test_injection, "gpu", "list of TensorGPU", lambda xs: [x._as_gpu() for x in xs]
    yield _test_injection, "cpu", "TensorListCPU", lambda xs: xs
    yield _test_injection, "gpu", "TensorListGPU", lambda xs: xs._as_gpu()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def es_pipeline_debug():
    images = fn.external_source(name="input")
    labels = fn.external_source(name="labels")
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, images, output, labels


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def es_pipeline_standard():
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, images, output, labels


def test_external_source_debug_sample_pipeline():
    n_iters = 10
    prefetch_queue_depth = 2
    pipe_load = load_images_pipeline()
    pipe_standard = es_pipeline_standard(prefetch_queue_depth=prefetch_queue_depth)
    pipe_debug = es_pipeline_debug(prefetch_queue_depth=prefetch_queue_depth)
    # Call feed_input `prefetch_queue_depth` extra times to avoid issues with
    # missing batches near the end of the epoch caused by prefetching
    for _ in range(n_iters + prefetch_queue_depth):
        images, labels = pipe_load.run()
        pipe_debug.feed_input("input", [np.array(t) for t in images])
        pipe_debug.feed_input("labels", np.array(labels.as_tensor()))
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def es_pipeline(source, batch):
    if source is not None:
        return fn.external_source(source, batch=batch, cycle=(not batch))
    else:
        return fn.external_source(name="input")


def _test_external_source_debug(source, batch):
    n_iters = 8
    prefetch_queue_depth = 2
    pipe_debug = es_pipeline(source, batch, prefetch_queue_depth=prefetch_queue_depth, debug=True)
    pipe_standard = es_pipeline(source, batch, prefetch_queue_depth=prefetch_queue_depth)
    if source is None:
        # Call feed_input `prefetch_queue_depth` extra times to avoid issues with
        # missing batches near the end of the epoch caused by prefetching
        for _ in range(n_iters + prefetch_queue_depth):
            x = np.random.rand(8, 5, 1)
            pipe_debug.feed_input("input", x)
            pipe_standard.feed_input("input", x)

    compare_pipelines(pipe_standard, pipe_debug, 8, n_iters)


def test_external_source_debug():
    for source in [np.random.rand(8, 8, 1), None]:
        for batch in [True, False]:
            yield _test_external_source_debug, source, batch


@pipeline_def(num_threads=3, device_id=0)
def es_pipeline_multiple_outputs(source, num_outputs):
    out1, out2, out3 = fn.external_source(source, num_outputs=num_outputs)
    return out1, out2, out3


def test_external_source_debug_multiple_outputs():
    n_iters = 13
    batch_size = 8
    num_outputs = 3
    data = [[np.random.rand(batch_size, 120, 120, 3)] * num_outputs] * n_iters
    pipe_debug = es_pipeline_multiple_outputs(data, num_outputs, batch_size=batch_size, debug=True)
    pipe_standard = es_pipeline_multiple_outputs(data, num_outputs, batch_size=batch_size)

    compare_pipelines(pipe_standard, pipe_debug, 8, n_iters)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def order_change_pipeline():
    if order_change_pipeline.change:
        rng = 0
    else:
        order_change_pipeline.change = True
        rng = fn.random.coin_flip(probability=0.5, seed=47)
    jpegs, labels = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        resized_images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, jpegs, labels, images, resized_images, output


@raises(
    RuntimeError,
    glob=(
        "Unexpected operator *. Debug mode does not support"
        " changing the order of operators executed within the pipeline."
    ),
)
def test_operators_order_change():
    order_change_pipeline.change = False
    pipe = order_change_pipeline()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def inputs_len_change():
    input = [np.zeros(1)] * 8
    if inputs_len_change.change:
        inputs_len_change.change = False
        inputs = [input]
    else:
        inputs = [input] * 2
    return fn.cat(*inputs)


@raises(
    RuntimeError,
    glob=("Trying to use operator * with different number of inputs than when" " it was built."),
)
def test_inputs_len_change():
    inputs_len_change.change = True
    pipe = inputs_len_change()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def kwargs_len_change():
    input = [np.zeros(1)] * 8
    inputs = [input] * 2
    kwargs = {}
    if kwargs_len_change.change:
        kwargs_len_change.change = False
        kwargs["axis"] = 0
    return fn.cat(*inputs, **kwargs)


@raises(
    RuntimeError,
    glob=(
        "Trying to use operator * with different number of keyword arguments"
        " than when it was built."
    ),
)
def test_kwargs_len_change():
    kwargs_len_change.change = True
    pipe = kwargs_len_change()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def inputs_batch_change():
    if inputs_batch_change.change:
        inputs_batch_change.change = False
        input = np.zeros(8)
    else:
        input = [np.zeros(1)] * 8
    return fn.random.coin_flip(input)


@raises(RuntimeError, glob="Input * for operator * is")
def test_inputs_batch_change():
    inputs_batch_change.change = True
    pipe = inputs_batch_change()
    pipe.run()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def kwargs_batch_change():
    kwargs = {}
    if kwargs_batch_change.change:
        kwargs_batch_change.change = False
        kwargs["probability"] = 0.75
    else:
        kwargs["probability"] = [np.zeros(1)] * 8
    return fn.random.coin_flip(**kwargs)


@raises(RuntimeError, glob="Argument * for operator * is")
def test_kwargs_batch_change():
    kwargs_batch_change.change = True
    pipe = kwargs_batch_change()
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


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def shape_pipeline(output_device):
    jpegs, _ = fn.readers.file(file_root=file_root, shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, device=output_device, output_type=types.RGB)
    assert images.shape() == [tuple(im.shape()) for im in images.get()]
    return images


def _test_shape_pipeline(device):
    pipe = shape_pipeline(device)
    (res,) = pipe.run()

    # Test TensorList.shape() directly.
    assert res.shape() == [tuple(im.shape()) for im in res]


def test_shape_pipeline():
    for device in ["cpu", "mixed"]:
        yield _test_shape_pipeline, device


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def seed_pipeline():
    coin_flip = fn.random.coin_flip()
    normal = fn.random.normal()
    uniform = fn.random.uniform()
    batch_permutation = fn.batch_permutation()
    return coin_flip, normal, uniform, batch_permutation


def test_seed_generation():
    pipe1 = seed_pipeline()
    pipe2 = seed_pipeline()
    compare_pipelines(pipe1, pipe2, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def seed_rn50_pipeline_base():
    rng = fn.random.coin_flip(probability=0.5)
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2, random_shuffle=True
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224))
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(
        resized_images.gpu(),
        mirror=rng,
        device="gpu",
        dtype=out_type,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return rng, jpegs, labels, images, resized_images, output


def test_seed_generation_base():
    pipe1 = seed_rn50_pipeline_base()
    pipe2 = seed_rn50_pipeline_base()
    compare_pipelines(pipe1, pipe2, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def device_change_rn50_pipeline_base():
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2, random_shuffle=True
    )
    images = fn.decoders.image(jpegs, output_type=types.RGB)

    if device_change_rn50_pipeline_base.change:
        images = images.gpu()

    output = fn.random_resized_crop(images, size=(224, 224))
    return labels, output


@raises(RuntimeError, glob="Input * for operator * is on * but was on * when created.")
def test_device_change():
    pipe = device_change_rn50_pipeline_base()
    device_change_rn50_pipeline_base.change = True
    pipe.run()
    device_change_rn50_pipeline_base.change = False
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def cpu_after_gpu_pipeline():
    jpegs, labels = fn.readers.file(
        file_root=file_root, shard_id=0, num_shards=2, random_shuffle=True
    )
    images = fn.decoders.image(jpegs, output_type=types.RGB, device="mixed")
    output = fn.random_resized_crop(images, size=(224, 224), device="cpu")
    return labels, output


@raises(RuntimeError, glob='incompatible device "gpu"')
def test_cpu_operator_after_gpu():
    pipe = cpu_after_gpu_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def input_sets_stateful_op_pipeline():
    set_size = 5
    jpegs = [
        fn.readers.file(file_root=file_root, seed=42, random_shuffle=True)[0]
        for _ in range(set_size)
    ]
    images = fn.decoders.image(jpegs, seed=42)
    output = fn.random_resized_crop(images, size=(224, 224), seed=42)

    assert len(output) == set_size
    return tuple(output)


def test_input_sets():
    pipe_standard = input_sets_stateful_op_pipeline()
    pipe_debug = input_sets_stateful_op_pipeline(debug=True)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def incorrect_input_sets_pipeline():
    jpegs, _ = fn.readers.file(file_root=file_root, seed=42, random_shuffle=True)
    images = fn.decoders.image(jpegs, seed=42)
    output = fn.cat([images, images, images], [images, images])

    return tuple(output)


@raises(
    ValueError,
    glob=(
        "All argument lists for Multiple Input Sets used with operator"
        " 'cat' must have the same length."
    ),
)
def test_incorrect_input_sets():
    pipe = incorrect_input_sets_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def multiple_input_sets_pipeline():
    jpegs = [
        fn.readers.file(file_root=file_root, seed=42, random_shuffle=True)[0] for _ in range(6)
    ]
    images = fn.decoders.image(jpegs, seed=42)
    cropped_images = fn.random_resized_crop(images, size=(224, 224), seed=42)
    output = fn.cat(cropped_images[:3], cropped_images[3:])
    return tuple(output)


def test_multiple_input_sets():
    pipe_standard = multiple_input_sets_pipeline()
    pipe_debug = multiple_input_sets_pipeline(debug=True)
    compare_pipelines(pipe_standard, pipe_debug, 8, 10)


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def variable_batch_size_from_external_source_pipeline(src_data):
    images = fn.external_source(src_data)
    output = fn.random_resized_crop(images, size=(32, 32))

    return (output,)


def test_variable_batch_size_from_external_source():
    batch_sizes = [3, 6, 7, 8]
    src_data = [np.zeros((batch_size, 64, 64, 3), dtype=np.uint8) for batch_size in batch_sizes]
    pipe = variable_batch_size_from_external_source_pipeline(src_data)
    for batch_size in batch_sizes:
        (output,) = pipe.run()
        assert len(output) == batch_size


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def incorrect_variable_batch_size_from_es_pipeline():
    rng = fn.random.coin_flip(probability=0.5)
    src_data = np.zeros((1, 6, 64, 64, 3), dtype=np.uint8)
    images = fn.external_source(src_data)
    return images, rng


@raises(
    RuntimeError,
    glob=(
        "Batch size must be uniform across an iteration."
        " External Source operator returned batch size*"
    ),
)
def test_incorrect_variable_batch_size_from_es():
    pipe = incorrect_variable_batch_size_from_es_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def incorrect_variable_batch_size_inside_es_pipeline():
    src_data = [
        [
            [np.ones((120, 120, 3), dtype=np.uint8)] * 8,
            [np.ones((120, 120, 3), dtype=np.float32)] * 6,
        ]
    ]
    out1, out2 = fn.external_source(
        source=src_data, num_outputs=2, dtype=[types.DALIDataType.UINT8, types.DALIDataType.FLOAT]
    )
    return out1, out2


@raises(RuntimeError, glob="External source must return outputs with consistent batch size.*")
def test_incorrect_variable_batch_size_inside_es():
    pipe = incorrect_variable_batch_size_inside_es_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def incorrect_variable_batch_size_pipeline():
    jpegs, labels = fn.readers.file(file_root=file_root)
    images = fn.decoders.image(jpegs)
    images = [images.get()[i] for i in range(6)]
    output = fn.random_resized_crop(images, size=(224, 224))
    return labels, output


@raises(RuntimeError, glob="Batch size must be uniform across an iteration. Input*")
def test_variable_batch_size():
    pipe = incorrect_variable_batch_size_pipeline()
    pipe.run()


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def unused_arg_es_pipeline(kwargs):
    return fn.external_source(np.zeros((2, 8, 1)), **kwargs)


def _test_es_unused_args(kwargs):
    pipe = unused_arg_es_pipeline(kwargs)
    pipe.run()


def test_external_source_unused_args():
    kwargs_list = [{"parallel": True}, {"foo": 123, "bar": "BAR"}]
    for kwargs in kwargs_list:
        yield _test_es_unused_args, kwargs


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def es_device_change_pipeline(source, device):
    return fn.external_source(source=source, device=device)


def _test_es_device_change(source, device):
    pipe = es_device_change_pipeline(source, device)
    (res,) = pipe.run()
    assert device in str(type(res)).lower()


def test_es_device_change():
    cpu_data = np.zeros((8, 1))
    gpu_data = tensors.TensorListCPU(cpu_data)._as_gpu()
    for data, device in zip([gpu_data], ["cpu"]):
        yield _test_es_device_change, data, device


@pipeline_def(batch_size=8, num_threads=3, device_id=0, seed=47, debug=True)
def nan_check_pipeline(source):
    return fn.constant(fdata=next(source), shape=())  # TODO: Constant handling in debug mode


def _test_nan_check(values):
    pipe = nan_check_pipeline(iter(values))
    for _ in range(2):
        pipe.run()


def test_nan_check():
    err_msg = "Argument 'fdata' for operator 'constant' unexpectedly changed value from*"
    for values in [[np.nan, 1], [1, np.nan]]:
        yield raises(RuntimeError, glob=err_msg)(_test_nan_check), values

    for values in [[1, 1], [np.nan, np.nan]]:
        yield _test_nan_check, values


def test_debug_pipeline_conditionals():
    @pipeline_def(batch_size=8, num_threads=3, device_id=0, enable_conditionals=False)
    def pipeline_split_merge():
        pred = fn.random.coin_flip(seed=42, dtype=types.BOOL)
        input = types.Constant(10, device="cpu")  # TODO: Consant handling in debug mode
        true, false = fn._conditional.split(input, predicate=pred)
        output_true = true + 2
        output_false = false + 100
        output = fn._conditional.merge(output_true, output_false, predicate=pred)
        print(
            f"Pred: {pred}, Output if: {output_true}, Output else: {output_false}, Output {output}"
        )
        return pred, output

    @pipeline_def(batch_size=8, num_threads=3, device_id=0, enable_conditionals=True)
    def pipeline_cond():
        pred = fn.random.coin_flip(seed=42, dtype=types.BOOL)
        input = types.Constant(10, device="cpu")  # TODO: Consant handling in debug mode
        print(f"Pred: {pred}")
        if pred:
            output = input + 2
            print(f"Output if: {output}")
        else:
            output = input + 100
            print(f"Output else: {output}")
        print(f"Output: {output}")
        return pred, output

    pipe_standard = pipeline_split_merge(debug=True)

    pipe_cond = pipeline_cond(debug=True)
    compare_pipelines(pipe_standard, pipe_cond, 8, 5)


def test_debug_pipeline_conditional_repeated_op():
    @pipeline_def(batch_size=8, num_threads=3, device_id=0, enable_conditionals=False)
    def pipeline_split_merge():
        pred = fn.random.coin_flip(seed=42, dtype=types.BOOL)
        rng1 = fn.random.coin_flip(seed=1)
        rng2 = fn.random.coin_flip(seed=2)
        true, _ = fn._conditional.split(rng1, predicate=pred)
        _, false = fn._conditional.split(rng2, predicate=pred)
        output_true = true + 20
        output_false = false + 10
        output = fn._conditional.merge(output_true, output_false, predicate=pred)
        print(
            f"Pred: {pred}, Output if: {output_true}, Output else: {output_false}, Output {output}"
        )
        return pred, output

    @pipeline_def(batch_size=8, num_threads=3, device_id=0, enable_conditionals=True)
    def pipeline_cond():
        pred = fn.random.coin_flip(seed=42, dtype=types.BOOL)
        rng1 = fn.random.coin_flip(seed=1)
        rng2 = fn.random.coin_flip(seed=2)
        print(f"Pred: {pred}")
        if pred:
            output = rng1 + 20
            print(f"Output if: {output}")
        else:
            output = rng2 + 10
            print(f"Output else: {output}")
        print(f"Output: {output}")
        return pred, output

    pipe_standard = pipeline_split_merge(debug=True)

    pipe_cond = pipeline_cond(debug=True)
    compare_pipelines(pipe_standard, pipe_cond, 8, 5)


def test_against_split_merge():
    for base_debug, conditional_debug in [(True, False), (False, True), (True, True)]:
        yield _impl_against_split_merge, {"debug": base_debug}, {"debug": conditional_debug}


def test_dot_gpu():
    for base_debug, conditional_debug in [(True, False), (False, True), (True, True)]:
        yield _impl_dot_gpu, {"debug": base_debug}, {"debug": conditional_debug}


def test_arg_inputs_scoped_tracking():
    for global_debug, scoped_debug in [(True, False), (False, True), (True, True)]:
        yield _impl_arg_inputs_scoped_tracking, {"debug": global_debug}, {"debug": scoped_debug}


def test_arg_inputs_scoped_uninitialized():
    yield _impl_arg_inputs_scoped_uninitialized, {"debug": True}


def test_generators():
    for pred in pred_gens[:-1]:
        for base_debug, conditional_debug in [(True, False), (False, True), (True, True)]:
            yield _impl_generators, pred, {"debug": base_debug}, {"debug": conditional_debug}


def test_uninitialized():
    yield _impl_uninitialized, {"debug": True}


def test_debug_pipeline_conditional_without_data_node():
    @pipeline_def(batch_size=8, num_threads=3, device_id=0, enable_conditionals=True)
    def pipeline_cond():
        pred = fn.random.coin_flip(seed=42, dtype=types.BOOL)
        rng1 = fn.random.coin_flip(seed=1)
        if pred:
            output = fn.copy(rng1.get())
        else:
            output = rng1 + 10
        return pred, output

    with assert_raises(
        ValueError,
        glob=(
            "Debug mode with conditional execution (when "
            "`enable_conditionals=True`) doesn't allow for modification of"
            " operator outputs by libraries other than DALI or using the"
            " TensorLists extracted via `.get()` as inputs."
            " Expected `DataNodeDebug` as an input, got * at input *."
        ),
    ):
        pipe_cond = pipeline_cond(debug=True)
        pipe_cond.run()
