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

import numpy as np
import nvidia.dali.backend_impl as _backend
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import eval_modes
from nose_utils import SkipTest, assert_raises, assert_warns
from nvidia.dali.experimental.dynamic._compile import CompiledBatch
from test_utils import get_dali_extra_path

dali_extra_path = get_dali_extra_path()
images_root = os.path.join(dali_extra_path, "db", "single", "jpeg")


def _is_compiled(batch):
    return isinstance(batch, CompiledBatch)


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
        # fmt: off
        images = ndd.crop_mirror_normalize(images, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=ndd.float32)  # noqa: E501
        # fmt: on
        compiled_results.append(ndd.as_tensor(images))
        assert _is_compiled(images)

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
        np.testing.assert_array_almost_equal(dyn, comp)


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

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
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
        assert len(compiled_results) == len(dynamic_results)
        for dyn, comp in zip(dynamic_results, compiled_results):
            np.testing.assert_array_equal(dyn, comp)


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

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
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

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
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

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
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


def test_compile_tensor_args():
    def make_reader():
        video_root = os.path.join(dali_extra_path, "db", "video", "sintel", "video_files")
        sequence_length = 60
        width, height = 108, 192
        return ndd.readers.VideoResize(
            filenames=[os.path.join(video_root, "sintel_trailer-720p_3.mp4")],
            sequence_length=sequence_length,
            device="gpu",
            resize_x=ndd.tensor(width),
            resize_y=height,
            file_list_include_preceding_frame=True,
        )

    reader_dyn = make_reader()
    reader_comp = make_reader()

    dynamic_results = []
    for (videos,) in reader_dyn.next_epoch(batch_size=4):
        rotated = ndd.rotate(videos, angle=60)
        dynamic_results.append(ndd.as_tensor(rotated).cpu())

    compiled_results = []
    for (videos,) in reader_comp.next_epoch(batch_size=4, compile=True):
        rotated = ndd.rotate(videos, angle=60)
        assert _is_compiled(rotated)
        compiled_results.append(ndd.as_tensor(rotated).cpu())

    assert len(dynamic_results) == len(compiled_results)
    for dyn, comp in zip(dynamic_results, compiled_results):
        np.testing.assert_array_equal(dyn, comp)
