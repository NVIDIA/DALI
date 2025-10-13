# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os.path
from test_utils import check_batch
import PIL.Image

try:
    from PIL.Image.Resampling import NEAREST, BILINEAR, BICUBIC, LANCZOS
except Exception:
    # Deprecated import, needed for Python 3.6
    from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS


def init_video_data():
    batch_size = 2
    video_directory = os.path.join(
        os.environ["DALI_EXTRA_PATH"], "db", "video", "sintel", "video_files"
    )

    video_files = [os.path.join(video_directory, f) for f in sorted(os.listdir(video_directory))]

    video_pipe = dali.pipeline.Pipeline(batch_size, 3, 0, seed=16)
    with video_pipe:
        input = fn.readers.video(device="gpu", filenames=video_files, sequence_length=32, stride=5)
        video_pipe.set_outputs(input)

    out = video_pipe.run()
    in_seq = np.array(out[0].at(0).as_cpu())
    return in_seq


frames_fhwc = init_video_data()
frames_fchw = frames_fhwc.transpose([0, 3, 1, 2])


def GetSequences(channel_first, length, batch_size):
    """gets overlapping sequences, starting at iteration number"""
    source = frames_fchw if channel_first else frames_fhwc
    N = source.shape[0]

    def get_seq(id):
        ret = []
        for k in range(length):
            i = (id + k) % N
            ret.append(source[i])
        return np.array(ret)

    def get_batch(iter):
        return [get_seq(iter * batch_size + i) for i in range(batch_size)]

    return get_batch


resample_dali2pil = {
    types.INTERP_NN: NEAREST,
    types.INTERP_TRIANGULAR: BILINEAR,
    types.INTERP_CUBIC: BICUBIC,
    types.INTERP_LANCZOS3: LANCZOS,
}


def resize_PIL(channel_first, interp, w, h):
    pil_resample = resample_dali2pil[interp]

    def resize(input):
        num_frames = input.shape[0]
        out_seq = []
        for i in range(num_frames):
            frame = input[i]
            if channel_first:
                frame = frame.transpose([1, 2, 0])
            out_frame = PIL.Image.fromarray(frame).resize([w, h], resample=pil_resample)
            out_frame = np.array(out_frame)
            if channel_first:
                out_frame = out_frame.transpose([2, 0, 1])
            out_seq.append(out_frame)
        return np.array(out_seq)

    return resize


def create_ref_pipe(channel_first, seq_len, interp, dtype, w, h, batch_size=2):
    pipe = dali.pipeline.Pipeline(batch_size, 1, 0, 0, exec_async=False, exec_pipelined=False)
    with pipe:
        layout = "FCHW" if channel_first else "FHWC"
        ext = fn.external_source(GetSequences(channel_first, seq_len, batch_size), layout=layout)
        pil_resized = fn.python_function(
            ext, function=resize_PIL(channel_first, interp, w, h), batch_processing=False
        )
        if dtype is not None:  # unfortunately, PIL can't quite handle that
            pil_resized = fn.cast(pil_resized, dtype=dtype)
        pil_resized = fn.reshape(pil_resized, layout=layout)
        pipe.set_outputs(pil_resized)
    return pipe


def create_dali_pipe(channel_first, seq_len, interp, dtype, w, h, batch_size=2):
    pipe = dali.pipeline.Pipeline(batch_size, 1, 0, 0)
    with pipe:
        layout = "FCHW" if channel_first else "FHWC"
        ext = fn.external_source(GetSequences(channel_first, seq_len, batch_size), layout=layout)
        resize_cpu_out = fn.resize(
            ext, resize_x=w, resize_y=h, interp_type=interp, dtype=dtype, save_attrs=True
        )
        resize_gpu_out = fn.resize(
            ext.gpu(),
            resize_x=w,
            resize_y=h,
            interp_type=interp,
            minibatch_size=4,
            dtype=dtype,
            save_attrs=True,
        )
        dali_resized_cpu, size_cpu = resize_cpu_out
        dali_resized_gpu, size_gpu = resize_gpu_out
        # extract just HW part from the input shape
        ext_size = fn.slice(
            fn.cast(ext.shape(), dtype=types.INT32), 2 if channel_first else 1, 2, axes=[0]
        )
        pipe.set_outputs(dali_resized_cpu, dali_resized_gpu, ext_size, size_cpu, size_gpu)
    return pipe


def _test_resize(layout, interp, dtype, w, h):
    channel_first = layout == "FCHW"
    pipe_dali = create_dali_pipe(channel_first, 8, interp, dtype, w, h)
    pipe_ref = create_ref_pipe(channel_first, 8, interp, dtype, w, h)
    eps = 1e-2
    max_err = 6
    for iter in range(4):
        out_dali = pipe_dali.run()
        out_ref = pipe_ref.run()[0]
        dali_cpu = out_dali[0]
        dali_gpu = out_dali[1]
        if interp == types.INTERP_LANCZOS3:
            # PIL can't resize float data. Lanczos resampling generates overshoot which we have
            # to get rid of for the comparison to succeed.
            dali_cpu = [np.array(x).clip(0, 255) for x in dali_cpu]
            dali_gpu = [np.array(x).clip(0, 255) for x in dali_gpu.as_cpu()]
        else:
            dali_cpu = [np.array(x) for x in dali_cpu]
            dali_gpu = [np.array(x) for x in dali_gpu.as_cpu()]
        if channel_first:
            out_ref = [np.array(x)[:, :, 1:-1, 1:-1] for x in out_ref]
            dali_gpu = [x[:, :, 1:-1, 1:-1] for x in dali_gpu]
            dali_cpu = [x[:, :, 1:-1, 1:-1] for x in dali_cpu]
        else:
            out_ref = [np.array(x)[:, 1:-1, 1:-1, :] for x in out_ref]
            dali_gpu = [x[:, 1:-1, 1:-1, :] for x in dali_gpu]
            dali_cpu = [x[:, 1:-1, 1:-1, :] for x in dali_cpu]
        check_batch(dali_cpu, out_ref, 2, eps=eps, max_allowed_error=max_err)
        check_batch(dali_gpu, out_ref, 2, eps=eps, max_allowed_error=max_err)
        ext_size = out_dali[2]
        size_cpu = out_dali[3]
        size_gpu = out_dali[4]
        check_batch(ext_size, size_cpu, 2)
        check_batch(ext_size, size_gpu, 2)


def test_resize():
    channel_first = False
    for interp, w, h in [
        (types.INTERP_NN, 640, 480),
        (types.INTERP_TRIANGULAR, 100, 80),
        (types.INTERP_LANCZOS3, 200, 100),
    ]:
        for dtype in [None, types.UINT8, types.FLOAT]:
            layout = "FCHW" if channel_first else "FHWC"
            channel_first = not channel_first  # alternating pattern cuts number of cases by half
            yield _test_resize, layout, interp, dtype, w, h
