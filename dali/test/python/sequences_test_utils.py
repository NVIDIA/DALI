# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import numpy as np

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali import types
import nvidia.dali.tensors as _Tensors

from test_utils import get_dali_extra_path, check_batch


data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
vid_file = os.path.join(data_root, 'db', 'video',
                        'sintel', 'sintel_trailer-720p.mp4')


def as_batch(tensor):
    if isinstance(tensor, _Tensors.TensorListGPU):
        tensor = tensor.as_cpu()
    return [np.array(sample, dtype=types.to_numpy_type(sample.dtype)) for sample in tensor]


def dummy_source(batches):
    def inner():
        while True:
            for batch in batches:
                yield batch
    return inner


def unfold_batch(batch, num_expand):
    assert(num_expand >= 0)
    if num_expand == 0:
        return batch
    if num_expand > 1:
        batch = [sample.reshape((-1,) + sample.shape[num_expand:])
                 for sample in batch]
    return [frame for sample in batch for frame in sample]


def unfold_batches(batches, num_expand):
    return [unfold_batch(batch, num_expand) for batch in batches]


def get_layout_prefix_len(layout, prefix):
    for i, c in enumerate(layout):
        if c not in prefix:
            return i
    return len(layout)


def expand_arg(input_layout, num_expand, arg_has_frames, input_batch, arg_batch):
    assert(1 <= num_expand <= 2)
    assert(len(input_batch) == len(arg_batch))
    expanded_batch = []
    for input_sample, arg_sample in zip(input_batch, arg_batch):
        if not arg_has_frames or len(arg_sample) == 1:
            arg_sample = arg_sample if not arg_has_frames else arg_sample[0]
            num_frames = np.prod(input_sample.shape[:num_expand])
            expanded_batch.extend(arg_sample for _ in range(num_frames))
        else:
            frame_idx = input_layout[:num_expand].find("F")
            assert(frame_idx >= 0)
            assert(len(arg_sample) == input_sample.shape[frame_idx])
            if num_expand == 1:
                expanded_batch.extend(
                    arg_frame for arg_frame in arg_sample)
            else:
                channel_idx = 1 - frame_idx
                assert(input_layout[channel_idx] == "C")
                if channel_idx > frame_idx:
                    expanded_batch.extend(frame_arg for frame_arg in arg_sample for _ in range(
                        input_sample.shape[channel_idx]))
                else:
                    expanded_batch.extend(frame_arg for _ in range(
                        input_sample.shape[channel_idx]) for frame_arg in arg_sample)
    return expanded_batch


def expand_arg_input(input_data, input_layout, expand_channels, arg_data, arg_layout):
    num_expand = get_layout_prefix_len(
        input_layout, "FC" if expand_channels else "F")
    arg_has_frames = arg_layout and arg_layout[0] == "F"
    ret_arg_layout = arg_layout if not arg_has_frames else arg_layout[1:]
    assert(len(input_data) == len(arg_data))
    expanded = [expand_arg(input_layout, num_expand, arg_has_frames, input_batch, arg_batch)
                for input_batch, arg_batch in zip(input_data, arg_data)]
    return expanded, ret_arg_layout


@pipeline_def
def vid_pipeline(num_frames, width, height, seq_layout):
    vid, _ = fn.readers.video_resize(
        filenames=[vid_file],
        labels=[],
        name='video reader',
        sequence_length=num_frames,
        file_list_include_preceding_frame=True,
        device='gpu',
        seed=42,
        resize_x=width,
        resize_y=height)
    if seq_layout == "FCHW":
        vid = fn.transpose(vid, perm=[0, 3, 1, 2])
    elif seq_layout == "CFHW":
        vid = fn.transpose(vid, perm=[3, 0, 1, 2])
    else:
        assert(seq_layout == "FHWC")
    return vid


def vid_source(batch_size, num_batches, num_frames, width, height, seq_layout):
    pipe = vid_pipeline(
        num_threads=4, batch_size=batch_size, num_frames=num_frames,
        width=width, height=height, device_id=0, seq_layout=seq_layout,
        prefetch_queue_depth=1)
    pipe.build()
    batches = []
    for _ in range(num_batches):
        (pipe_out,) = pipe.run()
        batches.append(as_batch(pipe_out))
    return batches


def _test_video(device, expand_channels, operator_fn, fixed_params, input_params,
                input_layout, input_data, rng):
    num_iters = 3
    # compute the arguments data here so that the parameters passed to _test_video are
    # a bit more readable when printed by nose
    input_params = get_input_params_data(
        input_data, input_layout, input_params, rng)

    @pipeline_def
    def pipeline(input_data, input_layout, input_params):
        input = fn.external_source(
            source=dummy_source(input_data), layout=input_layout)
        if device == "gpu":
            input = input.gpu()
        arg_nodes = {
            arg_name: fn.external_source(
                source=dummy_source(arg_data), layout=arg_layout)
            for arg_name, (arg_data, arg_layout) in input_params}
        output = operator_fn(input, **fixed_params, **arg_nodes)
        return output

    max_batch_size = max(len(batch) for batch in input_data)

    seq_pipe = pipeline(input_data=input_data, input_layout=input_layout, input_params=input_params,
                        batch_size=max_batch_size, num_threads=4,
                        device_id=0)

    num_expand = get_layout_prefix_len(
        input_layout, "FC" if expand_channels else "F")
    unfolded_input = unfold_batches(input_data, num_expand)
    unfolded_input_layout = input_layout[num_expand:]
    expanded_params = [
        (arg_name, expand_arg_input(input_data, input_layout,
         expand_channels, arg_data, arg_layout))
        for arg_name, (arg_data, arg_layout) in input_params]
    max_uf_batch_size = max(len(batch) for batch in unfolded_input)
    baseline_pipe = pipeline(input_data=unfolded_input,
                             input_layout=unfolded_input_layout,
                             input_params=expanded_params,
                             batch_size=max_uf_batch_size, num_threads=4,
                             device_id=0)
    seq_pipe.build()
    baseline_pipe.build()

    for _ in range(num_iters):
        (seq_batch,) = seq_pipe.run()
        (baseline_batch,) = baseline_pipe.run()
        assert(seq_batch.layout()[num_expand:] == baseline_batch.layout())
        batch = unfold_batch(as_batch(seq_batch), num_expand)
        baseline_batch = as_batch(baseline_batch)
        assert(len(batch) == len(baseline_batch))
        check_batch(batch, baseline_batch, len(batch))


max_batch_size = 8
max_num_frames = 16


def get_video_input_cases(seq_layout, rng):
    cases = []
    lager = vid_source(max_batch_size, 1, max_num_frames, 512, 288, seq_layout)
    smaller = vid_source(max_batch_size, 2, max_num_frames,
                         384, 216, seq_layout)
    cases.append(smaller)
    samples = [sample for batch in [smaller[0], lager[0], smaller[1]]
               for sample in batch]
    rng.shuffle(samples)
    case2 = [samples[0:1], samples[1:1 + max_batch_size],
             samples[1 + max_batch_size:2 * max_batch_size],
             samples[2 * max_batch_size:3*max_batch_size]]
    cases.append(case2)
    frames_idx = seq_layout.find("F")
    if frames_idx == 0:
        case3 = [[sample[:rng.randint(1, sample.shape[0])]
                  for sample in batch] for batch in case2]
        cases.append(case3)
    return cases


def get_input_params_data(input_data, input_layout, input_params, rng):

    def get_input_arg_per_sample(param_cb):
        param = [[param_cb(rng) for _ in batch] for batch in input_data]
        return param, None

    def get_input_arg_per_frame(param_cb):
        def arg_for_sample(num_frames):
            if rng.randint(1, 4) == 1:
                return np.array([param_cb(rng)])
            return np.array([param_cb(rng) for _ in range(num_frames)])
        frame_idx = input_layout.find("F")
        param = [[arg_for_sample(sample.shape[frame_idx])
                 for sample in batch] for batch in input_data]
        sample_dim = len(param[0][0].shape)
        return param, "F" + "*" * (sample_dim - 1)

    return [(param_name, get_input_arg_per_frame(param_cb)) if is_per_frame else
            (param_name, get_input_arg_per_sample(param_cb))
            for param_name, param_cb, is_per_frame in input_params]


def video_suite_helper(ops_test_cases, expand_channels=False):
    rng = random.Random(42)
    layouts = ["FHWC", "FCHW"]
    if expand_channels:
        layouts.append("CFHW")
    input_cases = [
        (input_layout, input_data)
        for input_layout in layouts
        for input_data in get_video_input_cases(input_layout, rng)]
    for operator_fn, fixed_params, input_params in ops_test_cases:
        for device in ["cpu", "gpu"]:
            for (input_layout, input_data) in input_cases:
                yield _test_video, device, expand_channels, operator_fn, fixed_params, input_params,\
                    input_layout, input_data, rng
