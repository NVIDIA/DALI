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

import itertools
import os
import sys

import numpy as np
from ndd_utils import _is_compiled, eval_modes
from nose2.tools import params
from nose_utils import SkipTest, assert_raises, assert_warns
from test_utils import get_dali_extra_path

import nvidia.dali.backend_impl as _backend
import nvidia.dali.experimental.dynamic as ndd

dali_extra_path = get_dali_extra_path()
images_root = os.path.join(dali_extra_path, "db", "single", "jpeg")


def _assert_parity(expected, actual):
    """Assert two result lists match element-wise; lengths must be equal."""
    for e, a in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(e, a)


def test_compile_mode_stickiness():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4):
        break
    with assert_raises(RuntimeError, glob="*cannot switch to compiled mode*"):
        for _ in reader.next_epoch(batch_size=4, compile=True):
            break


def test_compile_mode_stickiness_reverse():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4, compile=True):
        break
    with assert_raises(RuntimeError, glob="*cannot switch to non-compiled mode*"):
        for _ in reader.next_epoch(batch_size=4):
            break


def test_compile_requires_batch_size():
    reader = ndd.readers.File(file_root=images_root)
    with assert_raises(ValueError, glob="*requires a non-None batch_size*"):
        for _ in reader.next_epoch(compile=True):
            break


@eval_modes()
def test_compile_basic_pipeline():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        images = ndd.resize(images, size=[64, 64])
        images = ndd.crop_mirror_normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=ndd.float32,
        )
        dynamic_results.append(ndd.as_tensor(images))
        assert not _is_compiled(images)

    compiled_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        images = ndd.resize(images, size=[64, 64])
        images = ndd.crop_mirror_normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=ndd.float32,
        )
        compiled_results.append(ndd.as_tensor(images))
        assert _is_compiled(images)

    _assert_parity(dynamic_results, compiled_results)


@eval_modes()
def test_compile_same_call_site():
    def flip(images):
        return ndd.flip(images, horizontal=1)

    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=2, compile=True):
        images = ndd.decoders.image(jpegs)
        flipped1 = flip(images)
        flipped2 = flip(flipped1)
        assert _is_compiled(flipped1)
        assert _is_compiled(flipped2)
        np.testing.assert_array_equal(
            ndd.as_tensor(flipped2, pad=True),
            ndd.as_tensor(images, pad=True),
        )
        np.testing.assert_array_equal(
            ndd.as_tensor(flipped1, pad=True),
            ndd.as_tensor(images.slice[:, ::-1, :], pad=True),
        )


@eval_modes()
def test_compile_different_ops_same_call_site():
    ops = [ndd.flip, ndd.sphere]

    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for op in ops:
            out = op(images)
            assert not _is_compiled(out)
            dynamic_results.append(ndd.as_tensor(out, pad=True))

    for _ in range(3):
        compiled_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
            images = ndd.decoders.image(jpegs)
            for op in ops:
                out = op(images)
                assert _is_compiled(out)
                compiled_results.append(ndd.as_tensor(out, pad=True))

        _assert_parity(dynamic_results, compiled_results)


@eval_modes()
def test_compile_partial():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        size = [64, 64]
        resized = ndd.resize(images, size=size)
        dynamic_results.append(ndd.as_tensor(resized))

    compiled_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        assert _is_compiled(images)
        size = [64, 64]
        resized = ndd.resize(images, size=size)
        assert not _is_compiled(resized)
        compiled_results.append(ndd.as_tensor(resized))

    _assert_parity(dynamic_results, compiled_results)


@eval_modes()
def test_compile_multi_epoch():
    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        dynamic_results.append(ndd.as_tensor(images, pad=True))

    for _ in range(3):
        compiled_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
            images = ndd.decoders.image(jpegs)
            assert _is_compiled(images)
            compiled_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, compiled_results)


def test_compile_shard_rotation():
    reader_dyn = ndd.readers.File(file_root=images_root, shard_id=0, num_shards=2)
    reader_comp = ndd.readers.File(file_root=images_root, shard_id=0, num_shards=2)
    num_epochs = 4

    dynamic_epochs = []
    for _ in range(num_epochs):
        epoch = []
        for jpegs, _ in reader_dyn.next_epoch(batch_size=2):
            epoch.extend(np.asarray(sample).tobytes() for sample in jpegs)
        dynamic_epochs.append(epoch)

    compiled_epochs = []
    for _ in range(num_epochs):
        epoch = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=2, compile=True):
            images = ndd.decoders.image(jpegs)
            assert _is_compiled(images)
            epoch.extend(np.asarray(sample).tobytes() for sample in jpegs)
        compiled_epochs.append(epoch)

    assert dynamic_epochs == compiled_epochs


@eval_modes()
def test_compile_loop_identical():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            resized = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(resized))

    compiled_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            resized = ndd.resize(images, size=[64, 64])
            assert _is_compiled(resized)
        compiled_results.append(ndd.as_tensor(resized))

    _assert_parity(dynamic_results, compiled_results)


@eval_modes()
def test_compile_loop_data_dependent():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            images = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(images))

    compiled_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        for i in range(3):
            images = ndd.resize(images, size=[64, 64])
            assert _is_compiled(images) == (i == 0)
        compiled_results.append(ndd.as_tensor(images))

    _assert_parity(dynamic_results, compiled_results)


def test_compile_empty_graph():
    reader = ndd.readers.File(file_root=images_root)
    with assert_warns(UserWarning, glob="*no operators were captured*"):
        for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
            jpegs.evaluate()
    # After empty graph, reader should be back to clean state, can be reused in default mode
    for _ in reader.next_epoch(batch_size=4):
        break


@eval_modes()
def test_compile_diverging_inputs():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for i, (jpegs, _) in enumerate(reader_dyn.next_epoch(batch_size=2)):
        images = ndd.decoders.image(jpegs)
        if i % 2 == 0:
            images = ndd.flip(images, horizontal=1)
        images = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(images))

    compiled_results = []
    for i, (jpegs, _) in enumerate(reader_comp.next_epoch(batch_size=2, compile=True)):
        images = ndd.decoders.image(jpegs)
        if i % 2 == 0:
            images = ndd.flip(images, horizontal=1)
        assert _is_compiled(images)
        images = ndd.resize(images, size=[64, 64])
        # resize uses the compiled result only when its input matches tracing (flip was called)
        assert _is_compiled(images) == (i % 2 == 0)
        compiled_results.append(ndd.as_tensor(images))

    _assert_parity(dynamic_results, compiled_results)


def test_compile_batch_size_change_between_epochs():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4, compile=True):
        break
    with assert_raises(ValueError, glob="*cannot change batch_size*"):
        for _ in reader.next_epoch(batch_size=8, compile=True):
            break


def test_compile_batch_size_op_mismatch():
    reader = ndd.readers.File(file_root=images_root)

    for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        ndd.resize(images, size=[64, 64])

    with assert_raises(RuntimeError, glob="*cannot change batch_size*"):
        for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
            ndd.resize(ndd.decoders.image(jpegs), size=[64, 64], batch_size=8)


def test_compile_device_op_mismatch():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for device mismatch test")
    reader = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    for epoch, device in enumerate([None, "gpu"]):
        raised = False
        for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
            images = ndd.decoders.image(jpegs)
            try:
                ndd.resize(images, size=[64, 64], device=device)
            except RuntimeError as e:
                assert epoch == 1
                assert "Cannot change device" in str(e)
                raised = True
                break
        if raised:
            break
    else:
        assert False, "RuntimeError not raised for device mismatch"


def test_compile_stale_batch():
    reader = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    prev = None
    for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        # Iteration 0: target = images (fresh), resize is traced
        # Iteration 1+: target = prev (stale), falls back to dynamic
        target = prev if prev is not None else images
        resized = ndd.resize(target, size=[64, 64])
        assert _is_compiled(resized) == (prev is None)
        prev = images


def _make_video_reader(**resize_args):
    video_root = os.path.join(dali_extra_path, "db", "video", "sintel", "video_files")
    return ndd.readers.VideoResize(
        filenames=[os.path.join(video_root, "sintel_trailer-720p_3.mp4")],
        sequence_length=60,
        device="gpu",
        file_list_include_preceding_frame=True,
        **resize_args,
    )


def _test_video_resize(**resize_args):
    reader_dyn = _make_video_reader(**resize_args)
    reader_comp = _make_video_reader(**resize_args)

    dynamic_results = []
    for _ in range(3):
        for (videos,) in reader_dyn.next_epoch(batch_size=4):
            rotated = ndd.rotate(videos, angle=60)
            dynamic_results.append(ndd.as_tensor(rotated).cpu())

    compiled_results = []
    for _ in range(3):
        for (videos,) in reader_comp.next_epoch(batch_size=4, compile=True):
            rotated = ndd.rotate(videos, angle=60)
            assert _is_compiled(rotated)
            compiled_results.append(ndd.as_tensor(rotated).cpu())

    _assert_parity(dynamic_results, compiled_results)


def test_compile_tensor_arg():
    _test_video_resize(size=ndd.tensor([192, 108]))


def test_compile_tensor_arg_external():
    _test_video_resize(size=np.array([192, 108]))


def test_compile_scalar_args():
    _test_video_resize(resize_x=ndd.tensor(108), resize_y=192)


def test_reader_constructor_promotes_0d_tensor_args_to_scalars():
    reader = ndd.readers.Numpy(
        files=["unused.npy"],
        roi_start=ndd.tensor(0),
        roi_shape=ndd.tensor([10]),
    )

    assert reader._init_args["roi_start"] == 0
    assert "roi_start" in reader._tensor_arg_names
    assert "roi_start" not in reader._raw_tensor_args
    assert "roi_shape" in reader._tensor_arg_names
    assert "roi_shape" in reader._raw_tensor_args


def test_compile_incompatible_kwarg_dtype():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4, compile=False):
        img = ndd.decoders.image(jpegs, device="gpu")
        resized = ndd.tensor_resize(img, sizes=ndd._shape(img))
        dynamic_results.append(ndd.as_tensor(resized, pad=True).cpu())

    compiled_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
        img = ndd.decoders.image(jpegs, device="gpu")
        resized = ndd.tensor_resize(
            img,
            sizes=ndd._shape(img),
        )
        assert _is_compiled(resized), resized
        compiled_results.append(ndd.as_tensor(resized, pad=True).cpu())

    _assert_parity(dynamic_results, compiled_results)


def test_compile_nested_calls():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
        resized = ndd.resize(ndd.decoders.image(jpegs), size=[64, 64])
        if sys.version_info >= (3, 11):
            # PEP 657 positions disambiguate inner vs outer call by exact span.
            assert _is_compiled(resized)
        else:
            # Two calls share lineno on 3.10; ambiguous, both fall back to dynamic.
            assert not _is_compiled(resized)


def test_compile_multiple_calls_per_line():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
        images = ndd.decoders.image(jpegs)
        # fmt: off
        a = ndd.flip(images, horizontal=1); b = ndd.flip(a, horizontal=1)  # noqa: E501,E702
        # fmt: on
        if sys.version_info >= (3, 11):
            assert _is_compiled(a)
            assert _is_compiled(b)
        else:
            assert not _is_compiled(a)
            assert not _is_compiled(b)

        np.testing.assert_array_equal(ndd.as_tensor(images, pad=True), ndd.as_tensor(b, pad=True))


def test_compile_multiline_nested_calls():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, compile=True):
        resized = ndd.resize(
            ndd.decoders.image(jpegs),
            size=[64, 64],
        )
        assert _is_compiled(resized)


def _es(n_batches: int, batch_size: int, dim=3, **kwargs):
    """Finite (or cycling) ExternalSource of `n_batches` batches, distinct value per sample."""
    batches = [
        ndd.batch(
            [np.full((dim,), b * batch_size + s, dtype=np.float32) for s in range(batch_size)]
        )
        for b in range(n_batches)
    ]
    return ndd.ExternalSource(batches, **kwargs)


def _const_es(sample, batch_size: int):
    """Infinite callable ExternalSource returning a batch of `sample` each call."""
    sample = np.asarray(sample, dtype=np.float32)
    return ndd.ExternalSource(lambda: ndd.batch([sample] * batch_size))


@eval_modes()
def test_compile_es_basic():
    es_dyn = _es(3, batch_size=4)
    es_comp = _es(3, batch_size=4)

    dynamic_results = []
    for _ in range(3):
        out = ndd.cast(es_dyn(), dtype=ndd.int32)
        assert not _is_compiled(out)
        dynamic_results.append(ndd.as_tensor(out))

    compiled_results = []
    for batch in es_comp.compiled(batch_size=4):
        assert _is_compiled(batch)
        out = ndd.cast(batch, dtype=ndd.int32)
        assert _is_compiled(out)
        compiled_results.append(ndd.as_tensor(out))

    assert len(compiled_results) == 3
    _assert_parity(dynamic_results, compiled_results)


def test_compile_es_cycle_raise():
    es_dyn = _es(2, cycle="raise", batch_size=4)
    es_comp = _es(2, cycle="raise", batch_size=4)

    expected = []
    try:
        while True:
            expected.append(ndd.as_tensor(ndd.cast(es_dyn(), dtype=ndd.int32)))
    except StopIteration:
        pass

    for _ in range(3):
        compiled_results = []
        for batch in es_comp.compiled(batch_size=4):
            assert _is_compiled(batch)
            compiled_results.append(ndd.as_tensor(ndd.cast(batch, dtype=ndd.int32)))

        assert len(compiled_results) == 2
        _assert_parity(expected, compiled_results)


def test_compile_es_cycle_no():
    es = _es(3, cycle="no", batch_size=4)
    batches = []
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.int32)
        assert _is_compiled(batch)
        batches.append(batch)
    assert len(batches) == 3

    # The source is exhausted, a subsequent epoch yields nothing.
    second = []
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.int32)
        second.append(batch)
    assert second == []


def test_compile_es_multi_output():
    data = [
        (
            ndd.batch([np.full((2,), float(i), np.float32)] * 4),
            ndd.batch([np.full((3,), float(i + 10), np.float32)] * 4),
        )
        for i in range(2)
    ]
    es = ndd.ExternalSource(data, num_outputs=2)
    count = 0
    for outputs in es.compiled(batch_size=4):
        assert isinstance(outputs, tuple) and len(outputs) == 2
        a, b = outputs
        assert _is_compiled(a) and _is_compiled(b)
        np.testing.assert_array_equal(ndd.as_tensor(a)[0], [count, count])
        np.testing.assert_array_equal(ndd.as_tensor(b)[0], [count + 10] * 3)
        ndd.cast(a, dtype=ndd.float32)
        ndd.cast(b, dtype=ndd.float32)
        count += 1
    assert count == 2


def test_compile_es_broadcast():
    es = ndd.ExternalSource(lambda: np.arange(3, dtype=np.float32))
    expected = np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3))
    count = 0
    for batch in es.compiled(batch_size=4):
        assert batch.batch_size == 4
        np.testing.assert_array_equal(ndd.as_tensor(batch), expected)
        ndd.cast(batch, dtype=ndd.float32)
        count += 1
        if count >= 2:  # check both the traced and a compiled batch
            break
    assert count == 2


def test_compile_es_layout_dtype():
    es = ndd.ExternalSource(
        lambda: (np.zeros((4, 4, 3), np.float32), np.zeros((4, 4), np.float32)),
        num_outputs=2,
        layout=["HWC", "HW"],
        dtype=[ndd.float32, ndd.int32],
    )
    count = 0
    for a, b in es.compiled(batch_size=4):
        assert a.layout == "HWC" and b.layout == "HW"
        assert a.dtype == ndd.float32 and b.dtype == ndd.int32
        ndd.cast(a, dtype=ndd.float32)
        ndd.cast(b, dtype=ndd.int32)
        count += 1
        if count >= 2:  # check both the traced and a compiled batch
            break
    assert count == 2


def test_compile_es_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU device needed for the test")
    es = ndd.ExternalSource(lambda: np.arange(3, dtype=np.float32), device="gpu")
    count = 0
    for batch in es.compiled(batch_size=4):
        assert batch.device == ndd.Device("gpu")
        ndd.cast(batch, dtype=ndd.float32)
        count += 1
        if count >= 2:
            break
    assert count == 2


def test_compile_es_empty_graph():
    es = _es(2, batch_size=4, cycle="raise")
    with assert_warns(UserWarning, glob="no operators were captured"):
        for batch in es.compiled(batch_size=4):
            batch.evaluate()
    # After an empty graph the source falls back to eager mode and stays reusable.
    out = es()
    assert not _is_compiled(out)


def test_compile_es_break_reuse():
    es = _const_es([3, 3], batch_size=4)
    for i, batch in enumerate(es.compiled(batch_size=4)):
        ndd.cast(batch, dtype=ndd.int32)
        if i == 1:  # i=0 traced, i=1 the first compiled batch
            break

    count = 0
    for batch in es.compiled(batch_size=4):
        assert _is_compiled(batch)
        ndd.cast(batch, dtype=ndd.int32)
        count += 1
        if count >= 3:
            break
    assert count == 3


def test_compile_es_break_during_tracing():
    es = _const_es([1, 1], batch_size=4)
    feeder = _const_es([2, 2], batch_size=4)
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(feeder(), dtype=ndd.float32)
        break

    # Both unbound and reusable, the feeder was not left locked to the abandoned context.
    assert not _is_compiled(feeder())
    assert not _is_compiled(es())


def test_compile_reader_feeder():
    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    size_dyn = _const_es([64, 64], batch_size=4)
    size_comp = _const_es([64, 64], batch_size=4)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        images = ndd.resize(images, size=size_dyn())
        dynamic_results.append(ndd.as_tensor(images, pad=True))

    for _ in range(3):
        compiled_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
            images = ndd.decoders.image(jpegs)
            size = size_comp()
            assert _is_compiled(size)
            images = ndd.resize(images, size=size)
            assert _is_compiled(images)
            compiled_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, compiled_results)


def test_compile_es_feeder():
    es_dyn = _es(3, batch_size=4)
    es_comp = _es(3, batch_size=4)
    other_dyn = _const_es([7, 8], batch_size=4)
    other_comp = _const_es([7, 8], batch_size=4)

    dyn_a, dyn_b = [], []
    for _ in range(3):
        dyn_a.append(ndd.as_tensor(ndd.cast(es_dyn(), dtype=ndd.int32)))
        dyn_b.append(ndd.as_tensor(ndd.cast(other_dyn(), dtype=ndd.int32)))

    comp_a, comp_b = [], []
    for batch in es_comp.compiled(batch_size=4):
        a = ndd.cast(batch, dtype=ndd.int32)
        b = ndd.cast(other_comp(), dtype=ndd.int32)
        assert _is_compiled(a) and _is_compiled(b)
        comp_a.append(ndd.as_tensor(a))
        comp_b.append(ndd.as_tensor(b))

    assert len(comp_a) == 3
    _assert_parity(dyn_a, comp_a)
    _assert_parity(dyn_b, comp_b)


def test_compile_feeder_broadcast():
    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    size_dyn = ndd.ExternalSource(lambda: np.array([64, 64], dtype=np.float32))
    size_comp = ndd.ExternalSource(lambda: np.array([64, 64], dtype=np.float32))

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        images = ndd.resize(images, size=size_dyn())  # type: ignore
        dynamic_results.append(ndd.as_tensor(images, pad=True))

    for _ in range(3):
        compiled_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, compile=True):
            images = ndd.decoders.image(jpegs)
            size = size_comp()
            assert _is_compiled(size) and size.batch_size == 4
            images = ndd.resize(images, size=size)
            assert _is_compiled(images)
            compiled_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, compiled_results)


def test_compile_feeder_coexhaust():
    # Root and a finite feeder of equal length end the epoch cleanly
    es = _es(3, batch_size=4)
    extra = _es(3, batch_size=4)
    count = 0
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(extra(), dtype=ndd.float32)
        count += 1
    assert count == 3


# Tests for compiled ExternalSource misuse


def _es_no_compiled_after_eager():
    es = _es(3, batch_size=4)
    es()  # eager use locks the instance to eager mode
    for _ in es.compiled(batch_size=4):
        pass


def _es_no_eager_while_compiled():
    es = _es(3, batch_size=4)
    es.compiled(batch_size=4)  # binds the instance to a compiled loop
    es()


def _es_no_self_read():
    es = _const_es([1, 1], batch_size=4)
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(es(), dtype=ndd.float32)  # the iterated source must be read via the loop var


def _es_role_lock():
    es = _const_es([1, 1], batch_size=4)
    es.compiled(batch_size=4)  # es now iterates its own loop
    other = _es(3, batch_size=4)
    for batch in other.compiled(batch_size=4):
        ndd.cast(es(), dtype=ndd.float32)  # es already iterates its own loop
        ndd.cast(batch, dtype=ndd.float32)


def _feeder_context_lock():
    co = _const_es([3, 3], batch_size=4)
    es1 = _es(3, batch_size=4)
    es2 = _es(3, batch_size=4)

    for batch in es1.compiled(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)
        ndd.cast(batch, dtype=ndd.float32)

    for batch in es2.compiled(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)  # already bound to es1's context
        ndd.cast(batch, dtype=ndd.float32)


def _feeder_no_source_after_eager():
    co = _const_es([1, 1], batch_size=4)
    co()  # eager use
    es = _es(3, batch_size=4)
    for batch in es.compiled(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)  # can't become a compiled feeder now
        ndd.cast(batch, dtype=ndd.float32)


@params(
    (_es_no_compiled_after_eager, "used eagerly"),
    (_es_no_eager_while_compiled, "already used in a compiled loop"),
    (_es_no_self_read, "used through .compiled()"),
    (_es_role_lock, "used through .compiled()"),
    (_feeder_context_lock, "different compile context"),
    (_feeder_no_source_after_eager, "used eagerly"),
)
def test_compile_es_role_errors(scenario, glob):
    with assert_raises(RuntimeError, glob=glob):
        scenario()


def _feeder_underrun():
    es = _const_es([1, 1], batch_size=4)
    short = _es(1, batch_size=4)  # a single batch, then exhausted -> underrun
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(short(), dtype=ndd.float32)


def _feeder_read_once():
    es = _const_es([1, 1], batch_size=4)
    co = _const_es([2, 2], batch_size=4)
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(co(), dtype=ndd.float32)
        ndd.cast(co(), dtype=ndd.float32)  # second read in the same step


def _feeder_must_be_consumed():
    es = _es(3, batch_size=4)
    co = _const_es([2, 2], batch_size=4)
    for i, batch in enumerate(es.compiled(batch_size=4)):
        ndd.cast(batch, dtype=ndd.float32)
        if i == 0:  # consumed during tracing, then skipped -> not consumed next step
            ndd.cast(co(), dtype=ndd.float32)


def _feeder_late():
    es = _es(3, batch_size=4)
    late = _const_es([5, 5], batch_size=4)
    for i, batch in enumerate(es.compiled(batch_size=4)):
        ndd.cast(batch, dtype=ndd.float32)
        if i > 0:  # first used only after the trace iteration
            ndd.cast(late(), dtype=ndd.float32)


@params(
    (_feeder_underrun, "exhausted"),
    (_feeder_read_once, "once per compiled step"),
    (_feeder_must_be_consumed, "not consumed"),
    (_feeder_late, "during tracing"),
)
def test_compile_feeder_errors(scenario, glob):
    with assert_raises(RuntimeError, glob=glob):
        scenario()


def _es_batch_size_trace():
    es = _es(3, batch_size=5)
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)


def _es_batch_size_runtime():
    counter = itertools.count()
    es = ndd.ExternalSource(
        lambda: ndd.batch([np.zeros(3, np.float32)] * (4 if next(counter) == 0 else 5))
    )
    for batch in es.compiled(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)


@params(
    (_es_batch_size_trace, ValueError, "batch size 4"),
    (_es_batch_size_runtime, (ValueError, RuntimeError), "batch size"),
)
def test_compile_es_batch_size_errors(scenario, exc, glob):
    with assert_raises(exc, glob=glob):
        scenario()
