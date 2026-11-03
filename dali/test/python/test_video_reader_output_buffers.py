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

"""The output buffers of fn.experimental.readers.video on the GPU.

The GPU reader decodes a batch straight into the buffer that it returns (no copy). These tests
compare it with the copying path (DALI_VIDEO_READER_COPY_OUTPUTS=1, read when the operator is
created) and check that a returned batch is never written to again, whatever the executor and
the depths of the pipeline's and the reader's queues.
"""

import contextlib
import glob
import hashlib
import os
import tempfile

import numpy as np
from nose2.tools import cartesian_params, params
from nvidia.dali import fn, pipeline_def
from test_utils import get_dali_extra_path

DALI_EXTRA = get_dali_extra_path()
# H.264, 1280x720, files of different lengths
CLIPS = sorted(glob.glob(os.path.join(DALI_EXTRA, "db/video/sintel/video_files/*.mp4")))
# VP9 at 3840x2160, 1920x1080 and 334x240 (a row of 1002 bytes)
RESOLUTIONS = sorted(glob.glob(os.path.join(DALI_EXTRA, "db/video_resolution/vp9/*/*.mp4")))
COPY_ENV = "DALI_VIDEO_READER_COPY_OUTPUTS"
PIPELINE_ARGS = ("batch_size", "exec_dynamic", "prefetch_queue_depth", "enable_checkpointing")


@contextlib.contextmanager
def copy_outputs(enabled):
    old = os.environ.pop(COPY_ENV, None)
    if enabled:
        os.environ[COPY_ENV] = "1"
    try:
        yield
    finally:
        os.environ.pop(COPY_ENV, None)
        if old is not None:
            os.environ[COPY_ENV] = old


def make_pipe(copy, downstream=False, checkpoint=None, reader_depth=1, **kwargs):
    """A pipeline with the GPU video reader; `kwargs` are the pipeline's and the reader's
    arguments (by default: the clips, sequences of 8 frames every 40, labels)."""
    pipe_args = dict(batch_size=4, exec_dynamic=True, prefetch_queue_depth=2)
    pipe_args.update({k: kwargs.pop(k) for k in PIPELINE_ARGS if k in kwargs})
    reader_args = dict(sequence_length=8, step=40)
    if "file_list" not in kwargs:
        reader_args["filenames"] = CLIPS
        reader_args["labels"] = list(range(len(kwargs.get("filenames", CLIPS))))
    reader_args.update(kwargs)

    @pipeline_def(num_threads=2, device_id=0, seed=1234, **pipe_args)
    def pipe():
        outs = fn.experimental.readers.video(
            device="gpu", name="Reader", prefetch_queue_depth=reader_depth, **reader_args
        )
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        if downstream:  # a GPU operator that reads the reader's output
            outs.append(fn.flip(outs[0], horizontal=1))
        return tuple(outs)

    with copy_outputs(copy):  # read when the operator is created
        p = pipe(checkpoint=checkpoint) if checkpoint else pipe()
        p.build()
    return p


def to_numpy(out):
    out = out.as_cpu() if hasattr(out, "as_cpu") else out
    return [np.array(out[i]) for i in range(len(out))]


def digest(out):
    h = hashlib.sha1()
    for a in to_numpy(out):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def run(p, iters):
    """All outputs of `iters` iterations as numpy arrays, with the source info of output 0."""
    res = []
    for _ in range(iters):
        outs = p.run()
        info = [outs[0][i].source_info() for i in range(len(outs[0]))]
        res.append(([to_numpy(o) for o in outs], info))
    return res


def compare(ref, out):
    assert len(ref) == len(out)
    for it, ((ref_outs, ref_info), (outs, info)) in enumerate(zip(ref, out)):
        assert ref_info == info, f"iteration {it}: source_info {info} != {ref_info}"
        assert len(ref_outs) == len(outs)
        for o, (r, x) in enumerate(zip(ref_outs, outs)):
            assert len(r) == len(x)
            for i, (a, b) in enumerate(zip(r, x)):
                assert a.dtype == b.dtype and a.shape == b.shape, (it, o, i, a.shape, b.shape)
                assert np.array_equal(a, b), f"iteration {it}, output {o}, sample {i} differ"


def check_same_as_copy(iters=6, **kwargs):
    """The reader returns the same as with DALI_VIDEO_READER_COPY_OUTPUTS=1; returns that."""
    ref = run(make_pipe(True, **kwargs), iters)
    out = run(make_pipe(False, **kwargs), iters)
    compare(ref, out)
    return out


@cartesian_params((True, False), (1, 2, 3), (1, 2))
def test_same_as_copy(exec_dynamic, depth, reader_depth):
    check_same_as_copy(
        exec_dynamic=exec_dynamic,
        prefetch_queue_depth=depth,
        reader_depth=reader_depth,
        enable_frame_num="sequence",
        enable_timestamps=True,
    )


@params(
    dict(enable_frame_num="scalar"),
    dict(pad_mode="constant", fill_value=[10, 20, 30], enable_frame_num="sequence", step=100),
    dict(pad_mode="edge", enable_frame_num="sequence", sequence_length=16, step=100),
    dict(uniform_sample=True, sequence_length=5, enable_frame_num="sequence"),
    dict(stride=3, enable_timestamps=True, random_shuffle=True),
)
def test_same_as_copy_reader_args(reader_args):
    check_same_as_copy(**reader_args)


def test_same_as_copy_file_list_roi():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "list.txt")
        with open(path, "w") as f:
            for i, clip in enumerate(CLIPS):
                f.write(f"{clip} {i} {5 + i} {60 + 7 * i}\n")
        check_same_as_copy(
            file_list=path,
            file_list_format="frames",
            enable_frame_num="sequence",
            enable_timestamps=True,
            pad_mode="edge",
            step=30,
        )


@cartesian_params((1, 3), (1, 2))
def test_same_as_copy_resolutions(depth, reader_depth):
    # batches mix the three resolutions
    check_same_as_copy(
        iters=8,
        filenames=RESOLUTIONS,
        labels=[0, 1, 2],
        sequence_length=3,
        step=7,
        prefetch_queue_depth=depth,
        reader_depth=reader_depth,
    )


@cartesian_params((False, True), (1, 3), (1, 2))
def test_held_outputs_unchanged(downstream, depth, reader_depth):
    """Batches kept by the caller are never written to by the reader again."""
    p = make_pipe(
        False,
        downstream=downstream,
        prefetch_queue_depth=depth,
        reader_depth=reader_depth,
        enable_frame_num="sequence",
    )
    held = []
    for _ in range(3):
        outs = p.run()
        held.append((outs, [digest(o) for o in outs]))
    for _ in range(12):  # wraps around every queue several times
        p.run()
    for it, (outs, digests) in enumerate(held):
        assert [digest(o) for o in outs] == digests, f"held iteration {it} changed"


def epoch_size(**kwargs):
    return sum(make_pipe(True, **kwargs).epoch_size().values())


def check_padding(out, video_ptrs):
    """Samples with the same label and frame numbers are equal, and no two share a buffer;
    returns the number of such repeated samples."""
    repeated = 0
    for (outs, _), ptrs in zip(out, video_ptrs):
        video, labels, frame_num = outs[0], outs[1], outs[2]
        assert len(set(ptrs)) == len(ptrs), f"samples share a buffer: {ptrs}"
        for i in range(1, len(video)):
            if np.array_equal(labels[i], labels[i - 1]) and np.array_equal(
                frame_num[i], frame_num[i - 1]
            ):
                assert np.array_equal(video[i], video[i - 1])
                repeated += 1
    return repeated


def run_with_ptrs(p, iters):
    res, ptrs = [], []
    for _ in range(iters):
        outs = p.run()
        ptrs.append([outs[0][i].data_ptr() for i in range(len(outs[0]))])
        info = [outs[0][i].source_info() for i in range(len(outs[0]))]
        res.append(([to_numpy(o) for o in outs], info))
    return res, ptrs


@cartesian_params((True, False), (1, 3), (1, 2))
def test_pad_last_batch(exec_dynamic, depth, reader_depth):
    """The padding of the last batch repeats the last sample, each copy in its own buffer."""
    args = dict(filenames=CLIPS[:2], sequence_length=8, step=100, enable_frame_num="sequence")
    n = epoch_size(**args)
    batch_size = next(b for b in (4, 3, 5, 7) if n % b)
    args.update(
        pad_last_batch=True,
        batch_size=batch_size,
        exec_dynamic=exec_dynamic,
        prefetch_queue_depth=depth,
        reader_depth=reader_depth,
    )
    iters = 2 * -(-n // batch_size)  # two epochs
    ref = run(make_pipe(True, **args), iters)
    out, ptrs = run_with_ptrs(make_pipe(False, **args), iters)
    compare(ref, out)
    assert check_padding(out, ptrs) > 0


@cartesian_params((1, 3), (1, 2))
def test_padded_batch_of_a_shard(depth, reader_depth):
    """A shard with fewer batches than the other one returns a whole batch of copies of its
    last sample: the reader's queue can hold the same sample in two batches."""
    batch_size = 2
    for step in (100, 90, 80, 70, 60, 50):
        args = dict(filenames=CLIPS[:1], sequence_length=8, step=step, enable_frame_num="scalar")
        n = epoch_size(**args)
        shards = (n - n // 2, n // 2)
        if -(-shards[0] // batch_size) != -(-shards[1] // batch_size):
            break
    else:
        raise AssertionError("no step gives shards with different numbers of batches")
    for shard_id in (0, 1):
        args.update(
            pad_last_batch=True,
            num_shards=2,
            shard_id=shard_id,
            batch_size=batch_size,
            prefetch_queue_depth=depth,
            reader_depth=reader_depth,
        )
        iters = 3 * -(-shards[0] // batch_size)  # three epochs
        ref = run(make_pipe(True, **args), iters)
        out, ptrs = run_with_ptrs(make_pipe(False, **args), iters)
        compare(ref, out)
        repeated = check_padding(out, ptrs)
        assert shard_id == 0 or repeated > 0


@cartesian_params(
    (1, 2),
)
def test_checkpoint_resume(reader_depth):
    args = dict(enable_checkpointing=True, random_shuffle=True, reader_depth=reader_depth)
    p = make_pipe(False, **args)
    for _ in range(3):
        p.run()
    cpt = p.checkpoint()
    expected = run(p, 4)
    compare(expected, run(make_pipe(False, checkpoint=cpt, **args), 4))
