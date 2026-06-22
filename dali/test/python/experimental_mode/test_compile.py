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

import os
import sys

import numpy as np
import nvidia.dali.backend_impl as _backend
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import _is_compiled, eval_modes
from nose_utils import SkipTest, assert_raises, assert_warns
from test_utils import get_dali_extra_path

dali_extra_path = get_dali_extra_path()
images_root = os.path.join(dali_extra_path, "db", "single", "jpeg")


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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

        for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
            np.testing.assert_array_equal(dyn, comp)


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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
        for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
            np.testing.assert_array_equal(dyn, comp)


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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
    for (videos,) in reader_dyn.next_epoch(batch_size=4):
        rotated = ndd.rotate(videos, angle=60)
        dynamic_results.append(ndd.as_tensor(rotated).cpu())

    compiled_results = []
    for (videos,) in reader_comp.next_epoch(batch_size=4, compile=True):
        rotated = ndd.rotate(videos, angle=60)
        assert _is_compiled(rotated)
        compiled_results.append(ndd.as_tensor(rotated).cpu())

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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
    assert "roi_start" not in reader._original_tensor_args
    assert "roi_shape" in reader._tensor_arg_names
    assert "roi_shape" in reader._raw_tensor_args
    assert "roi_shape" in reader._original_tensor_args


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

    for dyn, comp in zip(dynamic_results, compiled_results, strict=True):
        np.testing.assert_array_equal(dyn, comp)


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
