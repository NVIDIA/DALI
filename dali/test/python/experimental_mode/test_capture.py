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
from ndd_utils import _is_captured, eval_modes
from nose2.tools import params
from nose_utils import SkipTest, assert_raises, assert_warns
from test_utils import get_dali_extra_path

import nvidia.dali.backend_impl as _backend
import nvidia.dali.experimental.dynamic as ndd

dali_extra_path = get_dali_extra_path()
images_root = os.path.join(dali_extra_path, "db", "single", "jpeg")


def _assert_parity(expected, actual):
    if isinstance(expected, (list, tuple)):
        for e, a in zip(expected, actual, strict=True):
            _assert_parity(e, a)
    else:
        np.testing.assert_array_equal(expected, actual)


def test_capture_mode_stickiness():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4):
        break
    with assert_raises(RuntimeError, glob="*cannot switch to capture mode*"):
        for _ in reader.next_epoch(batch_size=4, capture=True):
            break


def test_capture_mode_stickiness_reverse():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4, capture=True):
        break
    with assert_raises(RuntimeError, glob="*cannot switch to eager mode*"):
        for _ in reader.next_epoch(batch_size=4):
            break


def test_capture_requires_batch_size():
    reader = ndd.readers.File(file_root=images_root)
    with assert_raises(ValueError, glob="*requires a non-None batch_size*"):
        for _ in reader.next_epoch(capture=True):
            break


@eval_modes()
def test_capture_basic_pipeline():
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
        assert not _is_captured(images)

    captured_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        images = ndd.resize(images, size=[64, 64])
        images = ndd.crop_mirror_normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=ndd.float32,
        )
        captured_results.append(ndd.as_tensor(images))
        assert _is_captured(images)

    _assert_parity(dynamic_results, captured_results)


@eval_modes()
def test_capture_same_call_site():
    def flip(images):
        return ndd.flip(images, horizontal=1)

    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=2, capture=True):
        images = ndd.decoders.image(jpegs)
        flipped1 = flip(images)
        flipped2 = flip(flipped1)
        assert _is_captured(flipped1)
        assert _is_captured(flipped2)
        np.testing.assert_array_equal(
            ndd.as_tensor(flipped2, pad=True),
            ndd.as_tensor(images, pad=True),
        )
        np.testing.assert_array_equal(
            ndd.as_tensor(flipped1, pad=True),
            ndd.as_tensor(images.slice[:, ::-1, :], pad=True),
        )


@eval_modes()
def test_capture_different_ops_same_call_site():
    ops = [ndd.flip, ndd.sphere]

    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for op in ops:
            out = op(images)
            assert not _is_captured(out)
            dynamic_results.append(ndd.as_tensor(out, pad=True))

    for _ in range(3):
        captured_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
            images = ndd.decoders.image(jpegs)
            for op in ops:
                out = op(images)
                assert _is_captured(out)
                captured_results.append(ndd.as_tensor(out, pad=True))

        _assert_parity(dynamic_results, captured_results)


def test_capture_warmup():
    reader = ndd.readers.File(file_root=images_root)

    def epoch():
        iterations = 0
        for jpegs, _ in reader.next_epoch(batch_size=2, capture=True):
            images = ndd.decoders.image(jpegs)
            assert _is_captured(images)
            iterations += 1
        assert iterations > 0

    epoch()
    for _ in range(2):
        epoch()


def test_capture_proof_frames():
    reader = ndd.readers.File(file_root=images_root)

    def transform(images, angle):
        fill_value = 0

        def apply():
            return ndd.rotate(images, angle=angle, fill_value=fill_value)

        def middle():
            return apply()

        return middle()

    for i, (jpegs, _) in enumerate(reader.next_epoch(batch_size=2, capture=True)):
        images = ndd.decoders.image(jpegs)
        # Distinct call sites for transform, same arguments.
        if i == 0:
            rotated = transform(images, 10)
        else:
            rotated = transform(images, 10)
        assert _is_captured(rotated) is (i == 0)


@eval_modes()
def test_capture_partial():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        size = [64, 64]
        resized = ndd.resize(images, size=size)
        dynamic_results.append(ndd.as_tensor(resized))

    captured_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        assert _is_captured(images)
        size = [64, 64]
        resized = ndd.resize(images, size=size)
        assert not _is_captured(resized)
        captured_results.append(ndd.as_tensor(resized))

    _assert_parity(dynamic_results, captured_results)


@eval_modes()
def test_capture_multi_epoch():
    reader_dyn = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    reader_comp = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        dynamic_results.append(ndd.as_tensor(images, pad=True))

    for _ in range(3):
        captured_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
            images = ndd.decoders.image(jpegs)
            assert _is_captured(images)
            captured_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, captured_results)


def test_capture_shard_rotation():
    reader_dyn = ndd.readers.File(file_root=images_root, shard_id=0, num_shards=2)
    reader_comp = ndd.readers.File(file_root=images_root, shard_id=0, num_shards=2)
    num_epochs = 4

    dynamic_epochs = []
    for _ in range(num_epochs):
        epoch = []
        for jpegs, _ in reader_dyn.next_epoch(batch_size=2):
            epoch.extend(np.asarray(sample).tobytes() for sample in jpegs)
        dynamic_epochs.append(epoch)

    captured_epochs = []
    for _ in range(num_epochs):
        epoch = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=2, capture=True):
            images = ndd.decoders.image(jpegs)
            assert _is_captured(images)
            epoch.extend(np.asarray(sample).tobytes() for sample in jpegs)
        captured_epochs.append(epoch)

    assert dynamic_epochs == captured_epochs


@eval_modes()
def test_capture_loop_identical():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            resized = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(resized))

    captured_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            resized = ndd.resize(images, size=[64, 64])
            assert _is_captured(resized)
        captured_results.append(ndd.as_tensor(resized))

    _assert_parity(dynamic_results, captured_results)


@eval_modes()
def test_capture_loop_data_dependent():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4):
        images = ndd.decoders.image(jpegs)
        for _ in range(3):
            images = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(images))

    captured_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        for i in range(3):
            images = ndd.resize(images, size=[64, 64])
            assert _is_captured(images) == (i == 0)
        captured_results.append(ndd.as_tensor(images))

    _assert_parity(dynamic_results, captured_results)


def test_capture_empty_graph():
    reader = ndd.readers.File(file_root=images_root)
    with assert_warns(UserWarning, glob="*no operators were captured*"):
        for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
            jpegs.evaluate()
    # After empty graph, reader should be back to clean state, can be reused in default mode
    for _ in reader.next_epoch(batch_size=4):
        break


@eval_modes()
def test_capture_diverging_inputs():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for i, (jpegs, _) in enumerate(reader_dyn.next_epoch(batch_size=2)):
        images = ndd.decoders.image(jpegs)
        if i % 2 == 0:
            images = ndd.flip(images, horizontal=1)
        images = ndd.resize(images, size=[64, 64])
        dynamic_results.append(ndd.as_tensor(images))

    captured_results = []
    for i, (jpegs, _) in enumerate(reader_comp.next_epoch(batch_size=2, capture=True)):
        images = ndd.decoders.image(jpegs)
        if i % 2 == 0:
            images = ndd.flip(images, horizontal=1)
        assert _is_captured(images)
        images = ndd.resize(images, size=[64, 64])
        # resize uses the captured result only when its input matches tracing (flip was called)
        assert _is_captured(images) == (i % 2 == 0)
        captured_results.append(ndd.as_tensor(images))

    _assert_parity(dynamic_results, captured_results)


def test_capture_batch_size_change_between_epochs():
    reader = ndd.readers.File(file_root=images_root)
    for _ in reader.next_epoch(batch_size=4, capture=True):
        break
    with assert_raises(ValueError, glob="*cannot change batch_size*"):
        for _ in reader.next_epoch(batch_size=8, capture=True):
            break


def test_capture_batch_size_op_mismatch():
    reader = ndd.readers.File(file_root=images_root)

    for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        ndd.resize(images, size=[64, 64])

    with assert_raises(RuntimeError, glob="cannot change batch size"):
        for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
            ndd.resize(ndd.decoders.image(jpegs), size=[64, 64], batch_size=8)


def test_capture_device_op_mismatch():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for device mismatch test")
    reader = ndd.readers.File(file_root=images_root, pad_last_batch=True)

    for epoch, device in enumerate([None, "gpu"]):
        raised = False
        for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
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


def test_capture_stale_batch():
    reader = ndd.readers.File(file_root=images_root, pad_last_batch=True)
    prev = None
    for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        # Iteration 0: target = images (fresh), resize is traced
        # Iteration 1+: target = prev (stale), falls back to dynamic
        target = prev if prev is not None else images
        resized = ndd.resize(target, size=[64, 64])
        assert _is_captured(resized) == (prev is None)
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

    captured_results = []
    for _ in range(3):
        for (videos,) in reader_comp.next_epoch(batch_size=4, capture=True):
            rotated = ndd.rotate(videos, angle=60)
            assert _is_captured(rotated)
            captured_results.append(ndd.as_tensor(rotated).cpu())

    _assert_parity(dynamic_results, captured_results)


def test_capture_tensor_arg():
    _test_video_resize(size=ndd.tensor([192, 108]))


def test_capture_tensor_arg_external():
    _test_video_resize(size=np.array([192, 108]))


def test_capture_scalar_args():
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


def test_capture_incompatible_kwarg_dtype():
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dynamic_results = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=4, capture=False):
        img = ndd.decoders.image(jpegs, device="gpu")
        resized = ndd.tensor_resize(img, sizes=ndd._shape(img))
        dynamic_results.append(ndd.as_tensor(resized, pad=True).cpu())

    captured_results = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
        img = ndd.decoders.image(jpegs, device="gpu")
        resized = ndd.tensor_resize(
            img,
            sizes=ndd._shape(img),
        )
        assert _is_captured(resized), resized
        captured_results.append(ndd.as_tensor(resized, pad=True).cpu())

    _assert_parity(dynamic_results, captured_results)


def test_capture_nested_calls():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
        resized = ndd.resize(ndd.decoders.image(jpegs), size=[64, 64])
        if sys.version_info >= (3, 11):
            # PEP 657 positions disambiguate inner vs outer call by exact span.
            assert _is_captured(resized)
        else:
            # Two calls share lineno on 3.10; ambiguous, both fall back to dynamic.
            assert not _is_captured(resized)


def test_capture_multiple_calls_per_line():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
        images = ndd.decoders.image(jpegs)
        # fmt: off
        a = ndd.flip(images, horizontal=1); b = ndd.flip(a, horizontal=1)  # noqa: E501,E702
        # fmt: on
        if sys.version_info >= (3, 11):
            assert _is_captured(a)
            assert _is_captured(b)
        else:
            assert not _is_captured(a)
            assert not _is_captured(b)

        np.testing.assert_array_equal(ndd.as_tensor(images, pad=True), ndd.as_tensor(b, pad=True))


def test_capture_multiline_nested_calls():
    reader = ndd.readers.File(file_root=images_root)
    for jpegs, _ in reader.next_epoch(batch_size=4, capture=True):
        resized = ndd.resize(
            ndd.decoders.image(jpegs),
            size=[64, 64],
        )
        assert _is_captured(resized)


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
def test_capture_es_basic():
    es_dyn = _es(3, batch_size=4)
    es_comp = _es(3, batch_size=4)

    dynamic_results = []
    for _ in range(3):
        out = ndd.cast(es_dyn(), dtype=ndd.int32)
        assert not _is_captured(out)
        dynamic_results.append(ndd.as_tensor(out))

    captured_results = []
    for batch in es_comp.captured(batch_size=4):
        assert _is_captured(batch)
        out = ndd.cast(batch, dtype=ndd.int32)
        assert _is_captured(out)
        captured_results.append(ndd.as_tensor(out))

    assert len(captured_results) == 3
    _assert_parity(dynamic_results, captured_results)


def test_capture_es_cycle_raise():
    es_dyn = _es(2, cycle="raise", batch_size=4)
    es_comp = _es(2, cycle="raise", batch_size=4)

    expected = []
    try:
        while True:
            expected.append(ndd.as_tensor(ndd.cast(es_dyn(), dtype=ndd.int32)))
    except StopIteration:
        pass

    for _ in range(3):
        captured_results = []
        for batch in es_comp.captured(batch_size=4):
            assert _is_captured(batch)
            captured_results.append(
                ndd.as_tensor(
                    ndd.cast(batch, dtype=ndd.int32),
                )
            )

        assert len(captured_results) == 2
        _assert_parity(expected, captured_results)


def test_capture_es_cycle_no():
    es = _es(3, cycle="no", batch_size=4)
    batches = []
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.int32)
        assert _is_captured(batch)
        batches.append(batch)
    assert len(batches) == 3

    # The source is exhausted, a subsequent epoch yields nothing.
    second = []
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.int32)
        second.append(batch)
    assert second == []


def test_capture_es_multi_output():
    data = [
        (
            ndd.batch([np.full((2,), float(i), np.float32)] * 4),
            ndd.batch([np.full((3,), float(i + 10), np.float32)] * 4),
        )
        for i in range(2)
    ]
    es = ndd.ExternalSource(data, num_outputs=2)
    count = 0
    for outputs in es.captured(batch_size=4):
        assert isinstance(outputs, tuple) and len(outputs) == 2
        a, b = outputs
        assert _is_captured(a) and _is_captured(b)
        np.testing.assert_array_equal(ndd.as_tensor(a)[0], [count, count])
        np.testing.assert_array_equal(ndd.as_tensor(b)[0], [count + 10] * 3)
        ndd.cast(a, dtype=ndd.float32)
        ndd.cast(b, dtype=ndd.float32)
        count += 1
    assert count == 2


def test_capture_es_broadcast():
    es = ndd.ExternalSource(lambda: np.arange(3, dtype=np.float32))
    expected = np.broadcast_to(np.arange(3, dtype=np.float32), (4, 3))
    count = 0
    for batch in es.captured(batch_size=4):
        assert batch.batch_size == 4
        np.testing.assert_array_equal(ndd.as_tensor(batch), expected)
        ndd.cast(batch, dtype=ndd.float32)
        count += 1
        if count >= 2:  # check both the traced and a pipeline batch
            break
    assert count == 2


def test_capture_es_layout_dtype():
    es = ndd.ExternalSource(
        lambda: (np.zeros((4, 4, 3), np.float32), np.zeros((4, 4), np.float32)),
        num_outputs=2,
        layout=["HWC", "HW"],
        dtype=[ndd.float32, ndd.int32],
    )
    count = 0
    for a, b in es.captured(batch_size=4):
        assert a.layout == "HWC" and b.layout == "HW"
        assert a.dtype == ndd.float32 and b.dtype == ndd.int32
        ndd.cast(a, dtype=ndd.float32)
        ndd.cast(b, dtype=ndd.int32)
        count += 1
        if count >= 2:  # check both the traced and a pipeline batch
            break
    assert count == 2


def test_capture_es_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU device needed for the test")
    es = ndd.ExternalSource(lambda: np.arange(3, dtype=np.float32), device="gpu")
    count = 0
    for batch in es.captured(batch_size=4):
        assert batch.device == ndd.Device("gpu")
        ndd.cast(batch, dtype=ndd.float32)
        count += 1
        if count >= 2:
            break
    assert count == 2


def test_capture_es_empty_graph():
    es = _es(2, batch_size=4, cycle="raise")
    with assert_warns(UserWarning, glob="no operators were captured"):
        for batch in es.captured(batch_size=4):
            batch.evaluate()
    # After an empty graph the source falls back to eager mode and stays reusable.
    out = es()
    assert not _is_captured(out)


def test_capture_es_break_reuse():
    es = _const_es([3, 3], batch_size=4)
    for i, batch in enumerate(es.captured(batch_size=4)):
        ndd.cast(batch, dtype=ndd.int32)
        if i == 1:  # i=0 traced, i=1 the first pipeline batch
            break

    count = 0
    for batch in es.captured(batch_size=4):
        assert _is_captured(batch)
        ndd.cast(batch, dtype=ndd.int32)
        count += 1
        if count >= 3:
            break
    assert count == 3


def test_capture_es_break_during_tracing():
    es = _const_es([1, 1], batch_size=4)
    feeder = _const_es([2, 2], batch_size=4)
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(feeder(), dtype=ndd.float32)
        break

    # Both unbound and reusable, the feeder was not left locked to the abandoned context.
    assert not _is_captured(feeder())
    assert not _is_captured(es())


def test_capture_reader_feeder():
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
        captured_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
            images = ndd.decoders.image(jpegs)
            size = size_comp()
            assert _is_captured(size)
            images = ndd.resize(images, size=size)
            assert _is_captured(images)
            captured_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, captured_results)


def test_capture_es_feeder():
    es_dyn = _es(3, batch_size=4)
    es_comp = _es(3, batch_size=4)
    other_dyn = _const_es([7, 8], batch_size=4)
    other_comp = _const_es([7, 8], batch_size=4)

    dyn_a, dyn_b = [], []
    for _ in range(3):
        dyn_a.append(ndd.as_tensor(ndd.cast(es_dyn(), dtype=ndd.int32)))
        dyn_b.append(ndd.as_tensor(ndd.cast(other_dyn(), dtype=ndd.int32)))

    comp_a, comp_b = [], []
    for batch in es_comp.captured(batch_size=4):
        other_batch = other_comp()
        a = ndd.cast(batch, dtype=ndd.int32)
        b = ndd.cast(other_batch, dtype=ndd.int32)
        assert _is_captured(a) and _is_captured(b)
        comp_a.append(ndd.as_tensor(a))
        comp_b.append(ndd.as_tensor(b))

    assert len(comp_a) == 3
    _assert_parity(dyn_a, comp_a)
    _assert_parity(dyn_b, comp_b)


def test_capture_feeder_broadcast():
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
        captured_results = []
        for jpegs, _ in reader_comp.next_epoch(batch_size=4, capture=True):
            images = ndd.decoders.image(jpegs)
            size = size_comp()
            assert _is_captured(size) and size.batch_size == 4
            images = ndd.resize(images, size=size)
            assert _is_captured(images)
            captured_results.append(ndd.as_tensor(images, pad=True))
        _assert_parity(dynamic_results, captured_results)


def test_capture_feeder_coexhaust():
    # Root and a finite feeder of equal length end the epoch cleanly
    es = _es(3, batch_size=4)
    extra = _es(3, batch_size=4)
    count = 0
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(extra(), dtype=ndd.float32)
        count += 1
    assert count == 3


# Tests for capture-mode ExternalSource misuse


def _es_no_captured_after_eager():
    es = _es(3, batch_size=4)
    es()  # eager use locks the instance to eager mode
    for _ in es.captured(batch_size=4):
        pass


def _es_no_eager_while_captured():
    es = _es(3, batch_size=4)
    es.captured(batch_size=4)  # binds the instance to a capture-mode loop
    es()


def _es_no_self_read():
    es = _const_es([1, 1], batch_size=4)
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(es(), dtype=ndd.float32)  # the iterated source must be read via the loop var


def _es_role_lock():
    es = _const_es([1, 1], batch_size=4)
    es.captured(batch_size=4)  # es now iterates its own loop
    other = _es(3, batch_size=4)
    for batch in other.captured(batch_size=4):
        ndd.cast(es(), dtype=ndd.float32)  # es already iterates its own loop
        ndd.cast(batch, dtype=ndd.float32)


def _feeder_context_lock():
    co = _const_es([3, 3], batch_size=4)
    es1 = _es(3, batch_size=4)
    es2 = _es(3, batch_size=4)

    for batch in es1.captured(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)
        ndd.cast(batch, dtype=ndd.float32)

    for batch in es2.captured(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)  # already bound to es1's context
        ndd.cast(batch, dtype=ndd.float32)


def _feeder_no_source_after_eager():
    co = _const_es([1, 1], batch_size=4)
    co()  # eager use
    es = _es(3, batch_size=4)
    for batch in es.captured(batch_size=4):
        ndd.cast(co(), dtype=ndd.float32)  # can't become a captured feeder now
        ndd.cast(batch, dtype=ndd.float32)


@params(
    (_es_no_captured_after_eager, "used eagerly"),
    (_es_no_eager_while_captured, "already used in a capture-mode loop"),
    (_es_no_self_read, "used through .captured()"),
    (_es_role_lock, "used through .captured()"),
    (_feeder_context_lock, "different capture context"),
    (_feeder_no_source_after_eager, "used eagerly"),
)
def test_capture_es_role_errors(scenario, glob):
    with assert_raises(RuntimeError, glob=glob):
        scenario()


def _feeder_underrun():
    es = _const_es([1, 1], batch_size=4)
    short = _es(1, batch_size=4)  # a single batch, then exhausted -> underrun
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(short(), dtype=ndd.float32)


def _feeder_read_once():
    es = _const_es([1, 1], batch_size=4)
    co = _const_es([2, 2], batch_size=4)
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)
        ndd.cast(co(), dtype=ndd.float32)
        ndd.cast(co(), dtype=ndd.float32)  # second read in the same step


def _feeder_must_be_consumed():
    es = _es(3, batch_size=4)
    co = _const_es([2, 2], batch_size=4)
    for i, batch in enumerate(es.captured(batch_size=4)):
        ndd.cast(batch, dtype=ndd.float32)
        if i == 0:  # consumed during tracing, then skipped -> not consumed next step
            ndd.cast(co(), dtype=ndd.float32)


def _feeder_late():
    es = _es(3, batch_size=4)
    late = _const_es([5, 5], batch_size=4)
    for i, batch in enumerate(es.captured(batch_size=4)):
        ndd.cast(batch, dtype=ndd.float32)
        if i > 0:  # first used only after the trace iteration
            ndd.cast(late(), dtype=ndd.float32)


@params(
    (_feeder_underrun, "exhausted"),
    (_feeder_read_once, "once per capture-mode step"),
    (_feeder_must_be_consumed, "not consumed"),
    (_feeder_late, "during tracing"),
)
def test_capture_feeder_errors(scenario, glob):
    with assert_raises(RuntimeError, glob=glob):
        scenario()


def _es_batch_size_trace():
    es = _es(3, batch_size=5)
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)


def _es_batch_size_runtime():
    counter = itertools.count()
    es = ndd.ExternalSource(
        lambda: ndd.batch([np.zeros(3, np.float32)] * (4 if next(counter) == 0 else 5))
    )
    for batch in es.captured(batch_size=4):
        ndd.cast(batch, dtype=ndd.float32)


@params(
    (_es_batch_size_trace, ValueError, "batch size 4"),
    (_es_batch_size_runtime, (ValueError, RuntimeError), "batch size"),
)
def test_capture_es_batch_size_errors(scenario, exc, glob):
    with assert_raises(exc, glob=glob):
        scenario()


# Tests for random operators


def _uniform(rng):
    return ndd.random.uniform(batch_size=4, range=[0.0, 1.0], shape=[5], rng=rng)


def _collect_outputs(body, *, capture, epochs=1, reader=None):
    """Drive `body(step)` over epochs of one reader, from a single call site.
    Returns whatever the bodies returned, as host tensors.
    """
    reader = reader or ndd.readers.File(file_root=images_root, pad_last_batch=True)
    values, step = [], 0
    for _ in range(epochs):
        for _ in reader.next_epoch(batch_size=4, capture=capture):
            result = body(step)
            if result is not None:
                outputs = result if isinstance(result, tuple) else (result,)
                values.append(tuple(ndd.as_tensor(x, device="cpu") for x in outputs))
            step += 1
    return values


def _collect_random(reader, op, rng, *, capture=False, epochs=1, observe_state=False):
    states: list[str] = []

    def body(_):
        result = op(rng)
        if capture:
            outputs = result if isinstance(result, tuple) else (result,)
            assert all(_is_captured(x) for x in outputs)
        if observe_state:
            states.append(str(rng.state))
        return result

    return _collect_outputs(body, capture=capture, epochs=epochs, reader=reader), states


def _assert_random_parity(op, *, epochs=1, observe_state=False):
    values = []
    states = []
    for captured in (False, True):
        rng = ndd.random.RNG(seed=42)
        result, observed_states = _collect_random(
            ndd.readers.File(file_root=images_root, pad_last_batch=True),
            op,
            rng,
            capture=captured,
            epochs=epochs,
            observe_state=observe_state,
        )
        values.append(result)
        states.append((observed_states, str(rng.state)))

    _assert_parity(*values)
    assert states[0] == states[1]


def test_capture_random_parity():
    # Explicit rng
    _assert_random_parity(_uniform, epochs=3, observe_state=True)

    # Default rng
    ndd.random.set_seed(123)
    dynamic, _ = _collect_random(ndd.readers.File(file_root=images_root), _uniform, None)
    dynamic_state = str(ndd.random.get_default_rng().state)

    ndd.random.set_seed(123)
    captured, _ = _collect_random(
        ndd.readers.File(file_root=images_root), _uniform, None, capture=True
    )

    _assert_parity(dynamic, captured)
    assert dynamic_state == str(ndd.random.get_default_rng().state)


def test_capture_random_es_cycle_reset():
    dynamic_es = _es(2, cycle="raise", batch_size=4)
    dynamic_rng = ndd.random.RNG(seed=4)
    dynamic = []
    for _ in range(3):
        try:
            while True:
                dynamic_es()
                dynamic.append(ndd.as_tensor(_uniform(dynamic_rng), device="cpu"))
        except StopIteration:
            pass

    captured_es = _es(2, cycle="raise", batch_size=4)
    captured_rng = ndd.random.RNG(seed=4)
    captured = []
    for _ in range(3):
        for _ in captured_es.captured(batch_size=4):
            result = _uniform(captured_rng)
            assert _is_captured(result)
            captured.append(ndd.as_tensor(result, device="cpu"))

    _assert_parity(dynamic, captured)
    assert str(dynamic_rng.state) == str(captured_rng.state)


def test_capture_random_multiple_rngs():
    def rng_calls(rngs):
        rng, independent_rng = rngs
        first = ndd.random.uniform(batch_size=4, range=[0.0, 1.0], shape=[3], rng=rng)
        second = ndd.random.uniform(batch_size=4, range=[0.0, 1.0], shape=[3], rng=rng)
        independent = ndd.random.uniform(
            batch_size=4, range=[0.0, 1.0], shape=[3], rng=independent_rng
        )
        return first, second, independent

    dynamic_rngs = ndd.random.RNG(seed=8), ndd.random.RNG(seed=8)
    dynamic, _ = _collect_random(
        ndd.readers.File(file_root=images_root),
        rng_calls,
        dynamic_rngs,
        capture=False,
    )

    captured_rngs = ndd.random.RNG(seed=8), ndd.random.RNG(seed=8)
    captured, _ = _collect_random(
        ndd.readers.File(file_root=images_root),
        rng_calls,
        captured_rngs,
        capture=True,
    )

    _assert_parity(dynamic, captured)
    assert [str(rng.state) for rng in dynamic_rngs] == [str(rng.state) for rng in captured_rngs]


@params(False, True)
def test_capture_random_repeated_callsite(different_rngs):
    def make_rngs():
        first = ndd.random.RNG(seed=10)
        return (first, ndd.random.RNG(seed=11)) if different_rngs else (first, first)

    def body(rngs, batch):
        repeated = []
        for rng in rngs:
            x = ndd.random.uniform(batch_size=4, shape=[3], rng=rng)
            repeated.append(x)
        independent = ndd.cast(batch, dtype=ndd.float32)
        return (*repeated, independent)

    dynamic_rngs = make_rngs()
    dynamic = []
    dynamic_es = _es(3, batch_size=4)
    for _ in range(3):
        outputs = body(dynamic_rngs, dynamic_es())
        dynamic.append(tuple(ndd.as_tensor(output, device="cpu") for output in outputs))

    captured_rngs = make_rngs()
    captured = []
    captured_es = _es(3, batch_size=4)
    with assert_warns(UserWarning, glob="runs more than once per iteration"):
        for step, batch in enumerate(captured_es.captured(batch_size=4)):
            outputs = body(captured_rngs, batch)
            first, repeated, independent = outputs
            # The node carries one state, which the first call takes; only the repeat is eager.
            assert _is_captured(repeated) == (step == 0)
            for output in (first, independent):
                assert _is_captured(output)
            captured.append(tuple(ndd.as_tensor(output, device="cpu") for output in outputs))

    _assert_parity(dynamic, captured)
    assert [str(rng.state) for rng in dynamic_rngs] == [str(rng.state) for rng in captured_rngs]


def test_capture_random_shares_rng_with_uncapturable():
    dynamic_reader = ndd.readers.File(file_root=images_root)
    captured_reader = ndd.readers.File(file_root=images_root)

    def body(rngs, labels):
        shared_rng, other_rng = rngs
        independent = ndd.cast(labels, dtype=ndd.float32)
        early = ndd.random.uniform(batch_size=4, shape=[3], rng=other_rng)
        early_dependent = ndd.cast(early, dtype=ndd.float64)
        captured = ndd.random.uniform(batch_size=4, shape=[3], rng=shared_rng)
        bridge = ndd.noise.gaussian(captured, rng=other_rng)
        eager = ndd.random.uniform(shape=3, rng=shared_rng)  # no batch size, so not capturable
        return independent, early, early_dependent, captured, bridge, eager

    dynamic_rngs = ndd.random.RNG(seed=2), ndd.random.RNG(seed=3)
    dynamic = []
    for _, labels in dynamic_reader.next_epoch(batch_size=4):
        outputs = body(dynamic_rngs, labels)
        dynamic.append(tuple(ndd.as_tensor(x, device="cpu") for x in outputs))

    captured_rngs = ndd.random.RNG(seed=2), ndd.random.RNG(seed=3)
    captured = []
    for _, labels in captured_reader.next_epoch(batch_size=4, capture=True):
        outputs = body(captured_rngs, labels)
        *captured_outputs, eager = outputs

        # Only `eager` runs eagerly: sharing `shared_rng` with it costs others nothing
        for output in captured_outputs:
            assert _is_captured(output)
        assert isinstance(eager, ndd.Tensor)
        assert not _is_captured(eager)

        captured.append(tuple(ndd.as_tensor(x, device="cpu") for x in outputs))

    _assert_parity(dynamic, captured)
    assert [str(rng.state) for rng in dynamic_rngs] == [str(rng.state) for rng in captured_rngs]


def test_capture_random_rng_reuse():
    reference = ndd.random.RNG(seed=9)
    expected, _ = _collect_random(
        ndd.readers.File(file_root=images_root),
        _uniform,
        reference,
        epochs=2,
        capture=False,
    )

    rng = ndd.random.RNG(seed=9)
    first_reader = ndd.readers.File(file_root=images_root)
    first, _ = _collect_random(first_reader, _uniform, rng, capture=True)
    second, _ = _collect_random(
        ndd.readers.File(file_root=images_root),
        _uniform,
        rng,
        capture=True,
    )

    _assert_parity(expected, first + second)
    assert str(reference.state) == str(rng.state)
    with assert_raises(RuntimeError, glob="modified outside*loop"):
        _collect_random(first_reader, _uniform, rng, capture=True)


def test_capture_random_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for the GPU random test")
    _assert_random_parity(
        lambda rng: ndd.random.uniform(
            batch_size=4,
            range=[0.0, 1.0],
            shape=[5],
            rng=rng,
            device="gpu",
        )
    )


def _assert_body_parity(make_body, *, epochs=1, seed=11):
    """The same body in eager and capture modes: identical values and final RNG state."""
    runs = []
    for captured in (False, True):
        rng = ndd.random.RNG(seed=seed)
        runs.append(
            (_collect_outputs(make_body(rng), capture=captured, epochs=epochs), str(rng.state))
        )

    _assert_parity(runs[0][0], runs[1][0])
    assert runs[0][1] == runs[1][1]


@params("extra call after", "dropped call", "restore state")
def test_capture_random_contract_violation(kind):
    # Every way a body can draw differently than when it was traced
    rng = ndd.random.RNG(seed=21)

    def body(step):
        deviates = step > 0
        if deviates and kind == "restore state":
            rng.state = rng.state  # same position, but the generator was replaced

        _uniform(rng)

        if deviates and kind == "extra call after":
            ndd.random.uniform(shape=3, rng=rng)
        elif not deviates and kind == "dropped call":
            ndd.random.uniform(shape=3, rng=rng)  # traced, then never drawn again

    with assert_raises(RuntimeError, glob="used unexpectedly"):
        _collect_outputs(body, capture=True)


@params("bare before", "bare between")
def test_capture_random_untracked_draws(kind):
    # Draws the operator interception never sees still have to be accounted for
    def make_body(rng):
        def body(_):
            if kind == "bare before":
                rng()
            first = _uniform(rng)
            if kind == "bare between":
                rng()
                return first, _uniform(rng)
            return first

        return body

    _assert_body_parity(make_body)


class _AttributeArgBody:
    """A random operator the graph cannot record, because `self.prob` is an attribute."""

    def __init__(self, rng):
        self.rng = rng
        self.prob = 0.5

    def body(self, _):
        unrecordable = ndd.random.coin_flip(batch_size=4, probability=self.prob, rng=self.rng)
        return unrecordable, _uniform(self.rng)


def test_capture_random_shares_rng_with_unrecordable():
    _assert_body_parity(lambda rng: _AttributeArgBody(rng).body)

    seen = []
    holder = _AttributeArgBody(ndd.random.RNG(seed=11))
    _collect_outputs(
        lambda step: seen.append(tuple(map(_is_captured, holder.body(step)))), capture=True
    )
    assert seen and all(captured for _, captured in seen), seen
    assert not any(unrecordable for unrecordable, _ in seen), seen


def test_capture_random_repeated_callsite_first_falls_back():
    # A repeated site whose *first* call falls back on an input mismatch
    rng = ndd.random.RNG(seed=53)
    source_rng = ndd.random.RNG(seed=54)
    eager_input = ndd.random.uniform(batch_size=4, shape=[3], rng=ndd.random.RNG(seed=55))

    def body(step):
        captured_input = ndd.random.uniform(batch_size=4, shape=[3], rng=source_rng)
        inputs = [eager_input, captured_input] if step == 3 else [captured_input] * 2
        for source in inputs:  # one call site, two calls, fed differently on step 3
            ndd.noise.gaussian(source, rng=rng)

    with assert_warns(UserWarning, glob="runs more than once per iteration"):
        _collect_outputs(body, capture=True)


@params("advance", "seed")
def test_capture_random_touched_between_es_epochs(kind):
    # Recovery re-bases on the generator, so anything that moved it in between must be rejected
    es = _es(2, cycle="raise", batch_size=4)
    rng = ndd.random.RNG(seed=71)

    def epoch():
        for _ in es.captured(batch_size=4):
            _uniform(rng)

    epoch()
    if kind == "advance":
        rng.advance(4)
    else:
        rng.seed = 72

    with assert_raises(RuntimeError, glob="used unexpectedly"):
        epoch()


def test_capture_random_rng_reuse_one_period_gap():
    rng = ndd.random.RNG(seed=101)
    reader = ndd.readers.File(file_root=images_root)
    _collect_random(reader, _uniform, rng, capture=True)

    probe = ndd.random.RNG()
    _uniform(probe)  # one captured operator per iteration, so its draws are one period
    rng.advance(probe._draws)

    with assert_raises(RuntimeError, glob="used unexpectedly"):
        _collect_random(reader, _uniform, rng, capture=True)
