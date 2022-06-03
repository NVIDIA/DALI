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
from typing import List, Union, Callable
from dataclasses import dataclass

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali import types
import nvidia.dali.tensors as _Tensors

from test_utils import get_dali_extra_path, check_batch


data_root = get_dali_extra_path()
vid_file = os.path.join(data_root, 'db', 'video',
                        'sintel', 'sintel_trailer-720p.mp4')


@dataclass
class SampleDesc:
    """Context that the argument provider callback receives when prompted for parameter"""
    rng: random.Random
    frame_idx: int
    sample_idx: int
    batch_idx: int
    sample: np.ndarray


@dataclass
class ArgDesc:
    name: Union[str, int]
    is_per_frame: bool
    dest_device: str

    def __post_init__(self):
        assert self.is_positional_arg or self.dest_device == "cpu", "Named arguments on GPU are not supported"

    @property
    def is_positional_arg(self):
        return isinstance(self.name, int)


class ArgCb:
    """
    Describes a callback to be used as a per-sample/per-frame argument to the operator.
    ----------
    `name` : Union[str, int]
        String with the name of a named argument of the operator or an int if the data
             should be passed as a positional input.
    `cb` : Callable[[SampleDesc], np.ndarray]
        Callback that based on the SampleDesc instance produces a single parameter for specific sample/frame.
    `is_per_frame` : bool
        Flag if the cb should be run for every sample (sequence) or for every frame. In the latter case, the
        argument is passed wrapped in per-frame call to the operator.
    `dest_device` : str
        Controls whether the produced data should be passed to the operator in cpu or gpu memory.
        If set to "gpu", the copy to gpu is added in the pipeline. Applicable only to positional inputs.
    """

    def __init__(self, name: Union[str, int], cb: Callable[[SampleDesc], np.ndarray], is_per_frame: bool, dest_device: str = "cpu"):
        self.desc = ArgDesc(name, is_per_frame, dest_device)
        self.cb = cb

    def __repr__(self):
        return "ArgCb{}".format((self.cb, self.desc))


@dataclass
class ArgData:
    desc: ArgDesc
    data: List[List[np.ndarray]]


class ParamsProviderBase:
    """
    Computes data to be passed as argument inputs in sequence processing tests, the `compute_params` params
    should return a lists of ArgData; number of dimensions in the `arg_desc.data` must reflect
    `arg_data.desc.is_per_frame` argument.
    The `expand_params` should return corresponding unfolded/expanded ArgData to be used in the
    baseline pipeline.
    """

    def __init__(self):
        self.input_data = None
        self.input_layout = None
        self.fixed_params = None
        self.rng = None

        self.num_expand = None
        self.unfolded_input = None
        self.unfolded_input_layout = None

    def setup(self, input_data, input_layout, fixed_params, rng):
        self.input_data = input_data
        self.input_layout = input_layout
        self.fixed_params = fixed_params
        self.rng = rng

    def setup_expand(self, num_expand, unfolded_input, unfolded_input_layout):
        self.num_expand = num_expand
        self.unfolded_input = unfolded_input
        self.unfolded_input_layout = unfolded_input_layout

    def compute_params(self) -> List[ArgData]:
        raise NotImplementedError

    def expand_params(self) -> List[ArgData]:
        raise NotImplementedError


class ParamsProvider(ParamsProviderBase):

    def __init__(self, input_params: List[ArgCb]):
        super().__init__()
        self.input_params = input_params
        self.arg_input_data = None
        self.expanded_params_data = None

    def compute_params(self) -> List[ArgData]:
        self.arg_input_data = compute_input_params_data(
            self.input_data, self.input_layout, self.rng, self.input_params)
        return self.arg_input_data

    def expand_params(self) -> List[ArgData]:
        self.expanded_params_data = [
            ArgData(
                ArgDesc(arg_data.desc.name, False, arg_data.desc.dest_device),
                expand_arg_input(
                    self.input_data, self.input_layout, self.num_expand,
                    arg_data.data, arg_data.desc.is_per_frame))
            for arg_data in self.arg_input_data
        ]
        return self.expanded_params_data


def arg_data_node(arg_data: ArgData):
    node = fn.external_source(dummy_source(arg_data.data))
    if arg_data.desc.dest_device == "gpu":
        node = node.gpu()
    if arg_data.desc.is_per_frame:
        node = fn.per_frame(node)
    return node


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


def expand_arg_input(input_data, input_layout, num_expand, arg_data, arg_has_frames):
    assert(len(input_data) == len(arg_data))
    return [expand_arg(input_layout, num_expand, arg_has_frames, input_batch, arg_batch)
            for input_batch, arg_batch in zip(input_data, arg_data)]


def _test_seq_input(device, num_iters, expandable_extents, operator_fn, fixed_params, input_params,
                    input_layout, input_data, rng):

    @pipeline_def
    def pipeline(input_data, input_layout, args_data: List[ArgData]):
        input = fn.external_source(
            source=dummy_source(input_data), layout=input_layout)
        if device == "gpu":
            input = input.gpu()
        pos_args = [
            arg_data for arg_data in args_data if arg_data.desc.is_positional_arg]
        pos_nodes = [None] * (len(pos_args) + 1)
        for arg_data in pos_args:
            assert 0 <= arg_data.desc.name < len(pos_nodes)
            assert pos_nodes[arg_data.desc.name] is None
            pos_nodes[arg_data.desc.name] = arg_data_node(arg_data)
        [input_idx] = [i for i, pos_input in enumerate(
            pos_nodes) if pos_input is None]  # there should be exactly one
        pos_nodes[input_idx] = input
        named_args = [
            arg_data for arg_data in args_data if not arg_data.desc.is_positional_arg]
        arg_nodes = {
            arg_data.desc.name: arg_data_node(arg_data)
            for arg_data in named_args}
        output = operator_fn(*pos_nodes, **fixed_params, **arg_nodes)
        return output

    max_batch_size = max(len(batch) for batch in input_data)

    params_provider = input_params if isinstance(
        input_params, ParamsProviderBase) else ParamsProvider(input_params)
    params_provider.setup(input_data, input_layout, fixed_params, rng)
    args_data = params_provider.compute_params()
    seq_pipe = pipeline(input_data=input_data, input_layout=input_layout,
                        args_data=args_data,
                        batch_size=max_batch_size, num_threads=4,
                        device_id=0)

    num_expand = get_layout_prefix_len(input_layout, expandable_extents)
    unfolded_input = unfold_batches(input_data, num_expand)
    unfolded_input_layout = input_layout[num_expand:]
    params_provider.setup_expand(
        num_expand, unfolded_input, unfolded_input_layout)
    expanded_args_data = params_provider.expand_params()
    max_uf_batch_size = max(len(batch) for batch in unfolded_input)
    baseline_pipe = pipeline(input_data=unfolded_input,
                             input_layout=unfolded_input_layout,
                             args_data=expanded_args_data,
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


def get_input_arg_per_sample(input_data, param_cb, rng):
    return [[
        param_cb(SampleDesc(rng, None, sample_idx, batch_idx, sample))
        for sample_idx, sample in enumerate(batch)]
        for batch_idx, batch in enumerate(input_data)]


def get_input_arg_per_frame(input_data, input_layout, param_cb, rng, check_broadcasting):
    frame_idx = input_layout.find("F")

    def arg_for_sample(sample_idx, batch_idx, sample):
        if check_broadcasting and rng.randint(1, 4) == 1:
            return np.array([param_cb(SampleDesc(rng, 0, sample_idx, batch_idx, sample))])
        num_frames = sample.shape[frame_idx]
        return np.array([
            param_cb(SampleDesc(rng, frame_idx, sample_idx, batch_idx, sample))
            for frame_idx in range(num_frames)])

    return [[
        arg_for_sample(sample_idx, batch_idx, sample)
        for sample_idx, sample in enumerate(batch)]
        for batch_idx, batch in enumerate(input_data)]


def compute_input_params_data(input_data, input_layout, rng, input_params: List[ArgCb]):
    def input_param_data(arg_cb):
        if arg_cb.desc.is_per_frame:
            return get_input_arg_per_frame(
                input_data, input_layout, arg_cb.cb, rng, not arg_cb.desc.is_positional_arg)
        return get_input_arg_per_sample(input_data, arg_cb.cb, rng)
    return [ArgData(arg_cb.desc, input_param_data(arg_cb)) for arg_cb in input_params]


def sequence_suite_helper(rng, expandable_extents, input_cases, ops_test_cases, num_iters=4):
    class OpTestCase:
        def __init__(self, operator_fn, fixed_params, input_params, devices=None):
            self.operator_fn = operator_fn
            self.fixed_params = fixed_params
            self.input_params = input_params
            self.devices = ["cpu", "gpu"] if devices is None else devices
    for test_case_args in ops_test_cases:
        test_case = OpTestCase(*test_case_args)
        for device in test_case.devices:
            for (input_layout, input_data) in input_cases:
                yield _test_seq_input, device, num_iters, expandable_extents, \
                    test_case.operator_fn, test_case.fixed_params, \
                    test_case.input_params, input_layout, input_data, rng


def get_video_input_cases(seq_layout, rng, larger_shape=(512, 288), smaller_shape=(384, 216)):
    max_batch_size = 8
    max_num_frames = 16
    cases = []
    w, h = larger_shape
    larger = vid_source(max_batch_size, 1, max_num_frames,
                        w, h, seq_layout)
    w, h = smaller_shape
    smaller = vid_source(max_batch_size, 2, max_num_frames,
                         w, h, seq_layout)
    cases.append(smaller)
    samples = [sample for batch in [smaller[0], larger[0], smaller[1]]
               for sample in batch]
    rng.shuffle(samples)
    # test variable batch size
    case2 = [
        samples[0:1], samples[1:1 + max_batch_size],
        samples[1 + max_batch_size:2 * max_batch_size],
        samples[2 * max_batch_size:3 * max_batch_size]]
    cases.append(case2)
    frames_idx = seq_layout.find("F")
    if frames_idx == 0:
        # test variadic number of frames in different sequences
        case3 = [[sample[:rng.randint(1, sample.shape[0])]
                  for sample in batch] for batch in case2]
        cases.append(case3)
    return cases


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


def video_suite_helper(ops_test_cases, test_channel_first=True, expand_channels=False, rng=None):
    """
    Generates suite of video test cases for a sequence processing operator.
    The operator should meet the SequenceOperator assumptions, i.e.
    1. process frames (and possibly channels) independently,
    2. support per-frame tensor arguments.
    Each test case consists of two pipelines, one fed with the batch of sequences
    and one fed with the batch of frames, the test compares if the processing of
    corresponding frames in both pipelines gives the same result. In other words, if
    given batch = [sequence, ...], the following holds:
    fn.op([frame for sequence in batch for frame in sequence])
        == [frame for sequence in fn.op(batch) for frame in sequence]

    For testing operator with different input than the video, consider using `sequence_suite_helper` directly.
    ----------
    `ops_test_cases` : List[Tuple[Operator, Dict[str, Any], ParamProviderBase|List[ArgCb]]]
        List of operators and their parameters that should be tested.
        Each element is expected to be a triple of the form:
        [(fn.operator, {fixed_param_name: fixed_param_value}, [ArgCb(tensor_arg_name, single_arg_cb, is_per_frame, dest_device)])]
        where the first element is ``fn.operator``, the second one is a dictionary of fixed arguments that should
        be passed to the operator and the last one is a list of ArgCb instances describing tensor input arguments
        or custom params provider instance (see `ParamsProvider`).
    `test_channel_first` : bool
        If True, the "FCHW" layout is tested.
    `expand_channels` : bool
        If True, for the "FCHW" layout the first two (and not just one) dims are expanded, and "CFHW" layout is tested.
        Requires `test_channel_first` to be True.
    """
    rng = rng or random.Random(42)
    expandable_extents = "FC" if expand_channels else "F"
    layouts = ["FHWC"]
    if not test_channel_first:
        assert(not expand_channels)
    else:
        layouts.append("FCHW")
        if expand_channels:
            layouts.append("CFHW")
    input_cases = [
        (input_layout, input_data)
        for input_layout in layouts
        for input_data in get_video_input_cases(input_layout, rng)]
    yield from sequence_suite_helper(rng, expandable_extents, input_cases, ops_test_cases)
