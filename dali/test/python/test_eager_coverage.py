# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import io
import os
import re
from functools import reduce

from nvidia.dali import fn
from nvidia.dali import tensors
from nvidia.dali import types
from nvidia.dali.experimental import eager
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali._utils.eager_utils import _slice_tensorlist
from test_dali_cpu_only_utils import (
    pipeline_arithm_ops_cpu,
    setup_test_nemo_asr_reader_cpu,
    setup_test_numpy_reader_cpu,
)
from test_detection_pipeline import coco_anchors
from test_utils import check_batch, get_dali_extra_path, get_files, module_functions
from segmentation_test_utils import make_batch_select_masks
from webdataset_base import generate_temp_index_file as generate_temp_wds_index

""" Tests of coverage of eager operators. For each operator results from standard pipeline and
eager version are compared across a couple of iterations.
If you have added a new operator you should add a test here for an eager version of it. Also make
sure you have correctly classified the operator in `dali/python/nvidia/dali/_utils/eager_utils.py`
as stateless, stateful or iterator.
"""

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")
audio_files = get_files(os.path.join("db", "audio", "wav"), "wav")
caffe_dir = os.path.join(data_root, "db", "lmdb")
caffe2_dir = os.path.join(data_root, "db", "c2lmdb")
recordio_dir = os.path.join(data_root, "db", "recordio")
webdataset_dir = os.path.join(data_root, "db", "webdataset")
coco_dir = os.path.join(data_root, "db", "coco", "images")
coco_annotation = os.path.join(data_root, "db", "coco", "instances.json")
sequence_dir = os.path.join(data_root, "db", "sequence", "frames")
video_files = get_files(os.path.join("db", "video", "vfr"), "mp4")

rng = np.random.default_rng()

batch_size = 2
data_size = 10
sample_shape = [20, 20, 3]

# Sample data of image-like shape and type, used in many tests to avoid multiple object creation.
data = [
    [rng.integers(0, 255, size=sample_shape, dtype=np.uint8) for _ in range(batch_size)]
    for _ in range(data_size)
]

# Sample data for audio operators.
audio_data = [
    [rng.random(size=[200], dtype=np.float32) for _ in range(batch_size)] for _ in range(data_size)
]

# Sample data with single non-batch dimension.
flat_data = [
    [rng.integers(0, 255, size=[200], dtype=np.uint8) for _ in range(batch_size)]
    for _ in range(data_size)
]


def get_tl(data, layout="HWC"):
    """Utility function to create a TensorListCPU with given data and layout."""
    layout = "" if layout is None or (data.ndim != 4 and layout == "HWC") else layout
    return tensors.TensorListCPU(data, layout=layout)


def get_data(i):
    """Callback function to access data (numpy array) at given index. Used for generating inputs
    for standard pipelines.
    """
    return data[i]


def get_data_eager(i, layout="HWC"):
    """Callback function to access data at given index returned as TensorListCPU. Used for
    generating inputs for eager operators.
    """
    return get_tl(np.array(get_data(i)), layout)


def get_multi_data_eager(n):
    """Used for generating multiple inputs for eager operators."""

    def get(i, _):
        return tuple(get_data_eager(i) for _ in range(n))

    return get


class PipelineInput:
    """Class for generating inputs for pipeline.

    Args:
        pipe_fun: pipeline definition function.
        args: arguments for the pipeline creation.
        kwargs: possible keyword arguments used inside pipeline definition function.
    """

    def __init__(self, pipe_fun, *args, **kwargs) -> None:
        if kwargs:
            self.pipe = pipe_fun(*args, kwargs)
        else:
            self.pipe = pipe_fun(*args)

    def __call__(self, *_):
        return self.pipe.run()


class GetData:
    """Utility class implementing callback functions for pipeline and eager operators from
    a single dataset.
    """

    def __init__(self, data) -> None:
        """
        Construct the dataset

        Parameters
        ----------
        data : list of batches
            List of batches that will be returned as is or as TensorListCPU (depending on
            the fn or eager context).
        """
        self.data = data

    def fn_source(self, i):
        return self.data[i]

    def eager_source(self, i, layout="HWC"):
        return get_tl(np.array(self.fn_source(i)), layout)


def get_ops(op_path, fn_op=None, eager_op=None, eager_module=eager):
    """Get fn and eager versions of operators from given path."""

    import_path = op_path.split(".")
    if fn_op is None:
        fn_op = reduce(getattr, [fn] + import_path)
    if eager_op is None:
        eager_op = reduce(getattr, [eager_module] + import_path)
    return fn_op, eager_op


def compare_eager_with_pipeline(
    pipe,
    eager_op,
    *,
    eager_source=get_data_eager,
    layout="HWC",
    batch_size=batch_size,
    N_iterations=5,
    **kwargs,
):
    """Compares outputs from standard pipeline `pipe` and eager operator `eager_op` across
    `N_iterations`.
    """

    for i in range(N_iterations):
        input_tl = eager_source(i, layout)
        out_fn = pipe.run()
        if isinstance(input_tl, (tuple, list)):
            if len(input_tl):
                out_eager = eager_op(*input_tl, **kwargs)
            else:
                out_eager = eager_op(batch_size=batch_size, **kwargs)
        else:
            out_eager = eager_op(input_tl, **kwargs)

        if not isinstance(out_eager, (tuple, list)):
            out_eager = (out_eager,)

        assert len(out_fn) == len(out_eager)

        for tensor_out_fn, tensor_out_eager in zip(out_fn, out_eager):
            assert type(tensor_out_fn) is type(tensor_out_eager)

            if tensor_out_fn.dtype == types.BOOL:
                for t_fn, t_eager in zip(tensor_out_fn, tensor_out_eager):
                    assert np.array_equal(t_fn, t_eager)
            else:
                check_batch(tensor_out_fn, tensor_out_eager, batch_size)


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def single_op_pipeline(op, kwargs, source=get_data, layout="HWC"):
    data = fn.external_source(source=source, layout=layout)
    out = op(data, **kwargs)

    if isinstance(out, list):
        out = tuple(out)
    return out


def check_single_input(
    op_path,
    *,
    pipe_fun=single_op_pipeline,
    fn_source=get_data,
    fn_op=None,
    eager_source=get_data_eager,
    eager_op=None,
    layout="HWC",
    **kwargs,
):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = pipe_fun(fn_op, kwargs, source=fn_source, layout=layout)

    compare_eager_with_pipeline(pipe, eager_op, eager_source=eager_source, layout=layout, **kwargs)


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def no_input_pipeline(op, kwargs):
    out = op(**kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def no_input_source(*_):
    return ()


def check_no_input(
    op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=5, **kwargs
):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = no_input_pipeline(fn_op, kwargs)
    compare_eager_with_pipeline(
        pipe,
        eager_op,
        eager_source=no_input_source,
        batch_size=batch_size,
        N_iterations=N_iterations,
        **kwargs,
    )


def prep_stateful_operators(op_path):
    # Replicating seed that will be used inside rng_state, that way we expect fn and eager
    # operators to return same results.
    seed_upper_bound = (1 << 31) - 1
    seed = rng.integers(seed_upper_bound)
    fn_seed = np.random.default_rng(seed).integers(seed_upper_bound)
    eager_state = eager.rng_state(seed)

    fn_op, eager_op = get_ops(op_path, eager_module=eager_state)

    return fn_op, eager_op, fn_seed


def check_single_input_stateful(
    op_path,
    pipe_fun=single_op_pipeline,
    fn_source=get_data,
    fn_op=None,
    eager_source=get_data_eager,
    eager_op=None,
    layout="HWC",
    **kwargs,
):
    fn_op, eager_op, fn_seed = prep_stateful_operators(op_path)

    kwargs["seed"] = fn_seed
    pipe = pipe_fun(fn_op, kwargs, source=fn_source, layout=layout)
    kwargs.pop("seed", None)

    compare_eager_with_pipeline(pipe, eager_op, eager_source=eager_source, layout=layout, **kwargs)


def check_no_input_stateful(
    op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=5, **kwargs
):
    fn_op, eager_op, fn_seed = prep_stateful_operators(op_path)
    kwargs["seed"] = fn_seed
    pipe = no_input_pipeline(fn_op, kwargs)
    kwargs.pop("seed", None)
    compare_eager_with_pipeline(
        pipe,
        eager_op,
        eager_source=no_input_source,
        batch_size=batch_size,
        N_iterations=N_iterations,
        **kwargs,
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reader_pipeline(op, kwargs):
    out = op(pad_last_batch=True, **kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def check_reader(
    op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=2, **kwargs
):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = reader_pipeline(fn_op, kwargs)

    iter_eager = eager_op(batch_size=batch_size, **kwargs)

    for _ in range(N_iterations):
        for i, out_eager in enumerate(iter_eager):
            out_fn = pipe.run()

            if not isinstance(out_eager, (tuple, list)):
                out_eager = (out_eager,)

            assert len(out_fn) == len(out_eager)

            for tensor_out_fn, tensor_out_eager in zip(out_fn, out_eager):
                if i == len(iter_eager) - 1:
                    tensor_out_fn = _slice_tensorlist(tensor_out_fn, len(tensor_out_eager))

                assert type(tensor_out_fn) is type(tensor_out_eager)
                check_batch(tensor_out_fn, tensor_out_eager, len(tensor_out_eager))


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def file_reader_pipeline(kwargs):
    data, _ = fn.readers.file(**kwargs)
    return data


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reader_op_pipeline(op, kwargs, source=None, layout=None):
    if source is None:
        raise RuntimeError("No source for file reader.")
    data, _ = fn.readers.file(file_root=source)
    out = op(data, **kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def test_decoders_image():
    check_single_input(
        "decoders.image",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
    )


def test_experimental_decoders_image():
    check_single_input(
        "experimental.decoders.image",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
    )


def test_decoders_image_crop():
    check_single_input(
        "decoders.image_crop",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
        crop=(10, 10),
    )


def test_experimental_decoders_image_crop():
    check_single_input(
        "experimental.decoders.image_crop",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
        crop=(10, 10),
    )


def test_decoders_image_random_crop():
    check_single_input_stateful(
        "decoders.image_random_crop",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
    )


def test_experimental_decoders_image_random_crop():
    check_single_input_stateful(
        "experimental.decoders.image_random_crop",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
        output_type=types.RGB,
    )


def test_paste():
    check_single_input("paste", fill_value=0, ratio=2.0)


def test_decoders_numpy():
    def encode_sample(data):
        buff = io.BytesIO()
        np.save(buff, data)
        buff.seek(0)
        return np.frombuffer(buff.read(), dtype=np.uint8)

    in_data = [
        [encode_sample(np.arange(i * j).reshape(i, j)) for j in range(1, batch_size + 1)]
        for i in range(1, data_size + 1)
    ]
    eager_in_data = [tensors.TensorListCPU(batch) for batch in in_data]

    def fn_source(i):
        return in_data[i]

    def eager_source(i, layout):
        return eager_in_data[i]

    check_single_input(
        "decoders.numpy",
        fn_source=fn_source,
        eager_source=eager_source,
        layout=None,
    )


def test_rotate():
    check_single_input("rotate", angle=25)


def test_brightness_contrast():
    check_single_input("brightness_contrast")


def test_hue():
    check_single_input("hue")


def test_brightness():
    check_single_input("brightness")


def test_contrast():
    check_single_input("contrast")


def test_hsv():
    check_single_input("hsv")


def test_color_twist():
    check_single_input("color_twist")


def test_saturation():
    check_single_input("saturation")


def test__shape():
    check_single_input("_shape")


def test_crop():
    check_single_input("crop", crop=(5, 5))


def test_color_space_coversion():
    check_single_input("color_space_conversion", image_type=types.BGR, output_type=types.RGB)


def test_cast():
    check_single_input("cast", dtype=types.INT32)


def test_cast_like():
    source = np.array([1, 2, 3], dtype=np.int32)
    target = np.array([1.0], dtype=np.float32)

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
    def cast_like_pipe():
        return fn.cast_like(source, target)

    compare_eager_with_pipeline(
        cast_like_pipe(),
        lambda x: eager.cast_like(x, get_tl([target] * batch_size, None)),
        eager_source=lambda _i, _layout: get_tl([source] * batch_size, None),
        layout=None,
    )


def test_resize():
    check_single_input("resize", resize_x=50, resize_y=50)


def test_tensor_resize_cpu():
    check_single_input("experimental.tensor_resize", sizes=[50, 50], axes=[0, 1])


def test_per_frame():
    check_single_input("per_frame", replace=True)


def test_gaussian_blur():
    check_single_input("gaussian_blur", window_size=5)


def test_laplacian():
    check_single_input("laplacian", window_size=5)


def test_crop_mirror_normalize():
    check_single_input("crop_mirror_normalize")


def test_flip():
    check_single_input("flip", horizontal=True)


def test_jpeg_compression_distortion():
    check_single_input("jpeg_compression_distortion", quality=10)


def test_reshape():
    new_shape = sample_shape.copy()
    new_shape[0] //= 2
    new_shape[1] *= 2
    check_single_input("reshape", shape=new_shape)


def test_reinterpret():
    check_single_input("reinterpret", rel_shape=[0.5, 1, -1])


def test_water():
    check_single_input("water")


def test_sphere():
    check_single_input("sphere")


def test_erase():
    check_single_input(
        "erase",
        anchor=[0.3],
        axis_names="H",
        normalized_anchor=True,
        shape=[0.1],
        normalized_shape=True,
    )


def test_expand_dims():
    check_single_input("expand_dims", axes=1, new_axis_names="Z")


def test_coord_transform():
    M = [0, 0, 1, 0, 1, 0, 1, 0, 0]
    check_single_input("coord_transform", M=M, dtype=types.UINT8)


def test_grid_mask():
    check_single_input("grid_mask", tile=51, ratio=0.38158387, angle=2.6810782)


def test_multi_paste():
    check_single_input("multi_paste", in_ids=np.array([0, 1]), output_size=sample_shape)


def test_nonsilent_region():
    data = [
        [rng.integers(0, 255, size=[200], dtype=np.uint8) for _ in range(batch_size)]
    ] * data_size
    data[0][0][0] = 0
    data[0][1][0] = 0
    data[0][1][1] = 0
    get_data = GetData(data)

    check_single_input(
        "nonsilent_region",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout="",
    )


def test_preemphasis_filter():
    get_data = GetData(audio_data)
    check_single_input(
        "preemphasis_filter",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
    )


def test_power_spectrum():
    get_data = GetData(audio_data)
    check_single_input(
        "power_spectrum",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
    )


def test_spectrogram():
    get_data = GetData(audio_data)
    check_single_input(
        "spectrogram",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
        nfft=60,
        window_length=50,
        window_step=25,
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def mel_filter_pipeline(source):
    data = fn.external_source(source=source)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    processed = fn.mel_filter_bank(spectrum)
    return processed


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def mel_filter_input_pipeline(source):
    data = fn.external_source(source=source)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    return spectrum


def test_mel_filter_bank():
    compare_eager_with_pipeline(
        mel_filter_pipeline(audio_data),
        eager.mel_filter_bank,
        eager_source=PipelineInput(mel_filter_input_pipeline, audio_data),
    )


def test_to_decibels():
    get_data = GetData(audio_data)
    check_single_input(
        "to_decibels", fn_source=get_data.fn_source, eager_source=get_data.eager_source, layout=None
    )


def test_audio_resample():
    get_data = GetData(audio_data)
    check_single_input(
        "audio_resample",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
        scale=1.25,
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def mfcc_pipeline(source):
    data = fn.external_source(source=source)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    mel = fn.mel_filter_bank(spectrum)
    dec = fn.to_decibels(mel)
    processed = fn.mfcc(dec)

    return processed


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def mfcc_input_pipeline(source):
    data = fn.external_source(source=source)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    mel = fn.mel_filter_bank(spectrum)
    dec = fn.to_decibels(mel)

    return dec


def test_mfcc():
    compare_eager_with_pipeline(
        mfcc_pipeline(audio_data),
        eager.mfcc,
        eager_source=PipelineInput(mfcc_input_pipeline, audio_data),
    )


def test_one_hot():
    get_data = GetData(flat_data)
    check_single_input(
        "one_hot",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        num_classes=256,
        layout=None,
    )


def test_transpose():
    check_single_input("transpose", perm=[2, 0, 1])


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def audio_decoder_pipeline():
    data, _ = fn.readers.file(files=audio_files)
    out = fn.decoders.audio(data)
    return tuple(out)


def test_audio_decoder():
    compare_eager_with_pipeline(
        audio_decoder_pipeline(),
        eager.decoders.audio,
        eager_source=PipelineInput(file_reader_pipeline, files=audio_files),
    )


def test_coord_flip():
    get_data = GetData(
        [
            [
                (rng.integers(0, 255, size=[200, 2], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )

    check_single_input(
        "coord_flip", fn_source=get_data.fn_source, eager_source=get_data.eager_source, layout=None
    )


def test_bb_flip():
    get_data = GetData(
        [
            [
                (rng.integers(0, 255, size=[200, 4], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )

    check_single_input(
        "bb_flip", fn_source=get_data.fn_source, eager_source=get_data.eager_source, layout=None
    )


def test_warp_affine():
    check_single_input("warp_affine", matrix=(0.1, 0.9, 10, 0.8, -0.2, -20))


def test_warp_perspective():
    check_single_input("experimental.warp_perspective", matrix=np.eye(3))


def test_normalize():
    check_single_input("normalize")


def test_lookup_table():
    get_data = GetData(
        [
            [rng.integers(0, 5, size=[100], dtype=np.uint8) for _ in range(batch_size)]
            for _ in range(data_size)
        ]
    )

    check_single_input(
        "lookup_table",
        keys=[1, 3],
        values=[10, 50],
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def slice_pipeline(get_anchor, get_shape):
    data = fn.external_source(source=get_data, layout="HWC")
    anchors = fn.external_source(source=get_anchor)
    shape = fn.external_source(source=get_shape)
    processed = fn.slice(data, anchors, shape, out_of_bounds_policy="pad")

    return processed


def test_slice():
    get_anchors = GetData(
        [
            [
                (rng.integers(1, 256, size=[2], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )
    get_shapes = GetData(
        [
            [
                (rng.integers(1, 256, size=[2], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )

    def eager_source(i, _):
        return get_data_eager(i), get_anchors.eager_source(i), get_shapes.eager_source(i)

    pipe = slice_pipeline(get_anchors.fn_source, get_shapes.fn_source)
    compare_eager_with_pipeline(
        pipe, eager.slice, eager_source=eager_source, out_of_bounds_policy="pad"
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def image_decoder_slice_pipeline(get_anchors, get_shape):
    input, _ = fn.readers.file(file_root=images_dir)
    anchors = fn.external_source(source=get_anchors)
    shape = fn.external_source(source=get_shape)
    processed = fn.decoders.image_slice(input, anchors, shape)

    return processed


def test_image_decoder_slice():
    get_anchors = GetData(
        [
            [
                (rng.integers(1, 128, size=[2], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )
    get_shapes = GetData(
        [
            [
                (rng.integers(1, 128, size=[2], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )

    eager_input = file_reader_pipeline({"file_root": images_dir})

    def eager_source(i, _):
        return (
            eager_input.run()[0],
            get_anchors.eager_source(i, None),
            get_shapes.eager_source(i, None),
        )

    pipe = image_decoder_slice_pipeline(get_anchors.fn_source, get_shapes.fn_source)
    compare_eager_with_pipeline(pipe, eager.decoders.image_slice, eager_source=eager_source)


def test_pad():
    get_data = GetData(
        [
            [rng.integers(0, 255, size=[5, 4, 3], dtype=np.uint8) for _ in range(batch_size)]
            for _ in range(data_size)
        ]
    )

    check_single_input(
        "pad",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        fill_value=-1,
        axes=(0,),
        shape=(10,),
    )


def test_file_reader():
    check_reader("readers.file", file_root=images_dir)


def test_mxnet_reader():
    check_reader(
        "readers.mxnet",
        path=os.path.join(recordio_dir, "train.rec"),
        index_path=os.path.join(recordio_dir, "train.idx"),
        shard_id=0,
        num_shards=1,
    )


def test_webdataset_reader():
    webdataset = os.path.join(webdataset_dir, "MNIST", "devel-0.tar")
    webdataset_idx = generate_temp_wds_index(webdataset)
    check_reader(
        "readers.webdataset",
        paths=webdataset,
        index_paths=webdataset_idx.name,
        ext=["jpg", "cls"],
        shard_id=0,
        num_shards=1,
    )


def test_coco_reader():
    check_reader(
        "readers.coco",
        file_root=coco_dir,
        annotations_file=coco_annotation,
        shard_id=0,
        num_shards=1,
    )


def test_caffe_reader():
    check_reader("readers.caffe", path=caffe_dir, shard_id=0, num_shards=1)


def test_caffe2_reader():
    check_reader("readers.caffe2", path=caffe2_dir, shard_id=0, num_shards=1)


def test_nemo_asr_reader():
    tmp_dir, nemo_asr_manifest = setup_test_nemo_asr_reader_cpu()

    with tmp_dir:
        check_reader(
            "readers.nemo_asr",
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=False,
            read_sample_rate=True,
            read_text=True,
            seed=1234,
        )


def test_video_reader():
    check_reader("experimental.readers.video", filenames=video_files, sequence_length=3)


def test_copy():
    check_single_input("copy")


def test_element_extract():
    check_single_input("element_extract", element_map=[0, 3], layout=None)


def test_bbox_paste():
    get_data = GetData(
        [
            [
                (rng.integers(0, 255, size=[200, 4], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )
    check_single_input(
        "bbox_paste",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout=None,
        paste_x=0.25,
        paste_y=0.25,
        ratio=1.5,
    )


def test_sequence_rearrange():
    get_data = GetData(
        [
            [rng.integers(0, 255, size=[5, 10, 20, 3], dtype=np.uint8) for _ in range(batch_size)]
            for _ in range(data_size)
        ]
    )

    check_single_input(
        "sequence_rearrange",
        new_order=[0, 4, 1, 3, 2],
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        layout="FHWC",
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def box_encoder_pipeline(get_boxes, get_labels):
    boxes = fn.external_source(source=get_boxes)
    labels = fn.external_source(source=get_labels)
    out = fn.box_encoder(boxes, labels, anchors=coco_anchors())
    return tuple(out)


def test_box_encoder():
    get_boxes = GetData(
        [
            [
                (rng.integers(0, 255, size=[20, 4], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )
    get_labels = GetData(
        [
            [rng.integers(0, 255, size=[20, 1], dtype=np.int32) for _ in range(batch_size)]
            for _ in range(data_size)
        ]
    )

    def eager_source(i, _):
        return get_boxes.eager_source(i), get_labels.eager_source(i)

    pipe = box_encoder_pipeline(get_boxes.fn_source, get_labels.fn_source)
    compare_eager_with_pipeline(
        pipe, eager.box_encoder, eager_source=eager_source, anchors=coco_anchors()
    )


def test_numpy_reader():
    with setup_test_numpy_reader_cpu() as test_data_root:
        check_reader("readers.numpy", file_root=test_data_root)


def test_constant():
    check_no_input("constant", fdata=(1.25, 2.5, 3))


def test_dump_image():
    check_single_input("dump_image")


def test_affine_translate():
    check_no_input("transforms.translation", offset=(2, 3))


def test_affine_scale():
    check_no_input("transforms.scale", scale=(2, 3))


def test_affine_rotate():
    check_no_input("transforms.rotation", angle=30.0)


def test_affine_shear():
    check_no_input("transforms.shear", shear=(2.0, 1.0))


def test_affine_crop():
    check_no_input(
        "transforms.crop",
        from_start=(0.1, 0.2),
        from_end=(1.0, 1.2),
        to_start=(0.2, 0.3),
        to_end=(0.5, 0.6),
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def combine_transforms_pipeline():
    t = fn.transforms.translation(offset=(1, 2))
    r = fn.transforms.rotation(angle=30.0)
    s = fn.transforms.scale(scale=(2, 3))
    out = fn.transforms.combine(t, r, s)

    return out


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def combine_transforms_input_pipeline():
    t = fn.transforms.translation(offset=(1, 2))
    r = fn.transforms.rotation(angle=30.0)
    s = fn.transforms.scale(scale=(2, 3))

    return t, r, s


def test_combine_transforms():
    compare_eager_with_pipeline(
        combine_transforms_pipeline(),
        eager.transforms.combine,
        eager_source=PipelineInput(combine_transforms_input_pipeline),
    )


def test_reduce_min():
    check_single_input("reductions.min")


def test_reduce_max():
    check_single_input("reductions.max")


def test_reduce_sum():
    check_single_input("reductions.sum")


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def segmentation_select_masks_pipeline(source):
    device = "cpu" if Pipeline.current().device_id is None else "gpu"
    polygons, vertices, selected_masks = fn.external_source(
        source=source, num_outputs=3, device=device
    )
    out_polygons, out_vertices = fn.segmentation.select_masks(
        selected_masks, polygons, vertices, reindex_masks=False
    )

    return out_polygons, out_vertices


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def segmentation_select_masks_input_pipeline(source):
    device = "cpu" if Pipeline.current().device_id is None else "gpu"
    polygons, vertices, selected_masks = fn.external_source(
        source=source, num_outputs=3, device=device
    )

    return selected_masks, polygons, vertices


def test_segmentation_select_masks():
    data = [
        make_batch_select_masks(
            batch_size, vertex_ndim=2, npolygons_range=(1, 5), nvertices_range=(3, 10)
        )
        for _ in range(data_size)
    ]

    pipe = segmentation_select_masks_pipeline(data)
    compare_eager_with_pipeline(
        pipe,
        eager.segmentation.select_masks,
        eager_source=PipelineInput(segmentation_select_masks_input_pipeline, data),
    )


def test_reduce_mean():
    check_single_input("reductions.mean")


def test_reduce_mean_square():
    check_single_input("reductions.mean_square")


def test_reduce_root_mean_square():
    check_single_input("reductions.rms")


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reduce_pipeline(op):
    data = fn.external_source(source=get_data)
    mean = fn.reductions.mean(data)
    out = op(data, mean)

    return out


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reduce_input_pipeline():
    data = fn.external_source(source=get_data)
    mean = fn.reductions.mean(data)

    return data, mean


def test_reduce_std():
    pipe = reduce_pipeline(fn.reductions.std_dev)
    compare_eager_with_pipeline(
        pipe, eager_op=eager.reductions.std_dev, eager_source=PipelineInput(reduce_input_pipeline)
    )


def test_reduce_variance():
    pipe = reduce_pipeline(fn.reductions.variance)
    compare_eager_with_pipeline(
        pipe, eager_op=eager.reductions.variance, eager_source=PipelineInput(reduce_input_pipeline)
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def multi_input_pipeline(op, n):
    data = [fn.external_source(source=get_data, layout="HWC") for _ in range(n)]
    out = op(*data)

    return out


def test_cat():
    num_inputs = 3
    compare_eager_with_pipeline(
        multi_input_pipeline(fn.cat, num_inputs),
        eager_op=eager.cat,
        eager_source=get_multi_data_eager(num_inputs),
    )


def test_stack():
    num_inputs = 3
    compare_eager_with_pipeline(
        multi_input_pipeline(fn.stack, num_inputs),
        eager_op=eager.stack,
        eager_source=get_multi_data_eager(num_inputs),
    )


def test_batch_permute():
    check_single_input("permute_batch", indices=rng.permutation(batch_size).tolist())


def test_squeeze():
    get_data = GetData(
        [[np.zeros(shape=[10, 20, 3, 1, 1], dtype=np.uint8) for _ in range(batch_size)]] * data_size
    )
    check_single_input(
        "squeeze",
        fn_source=get_data.fn_source,
        eager_source=get_data.eager_source,
        axis_names="YZ",
        layout="HWCYZ",
    )


def test_peek_image_shape():
    check_single_input(
        "peek_image_shape",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
    )


def test_experimental_peek_image_shape():
    check_single_input(
        "experimental.peek_image_shape",
        pipe_fun=reader_op_pipeline,
        fn_source=images_dir,
        eager_source=PipelineInput(file_reader_pipeline, file_root=images_dir),
    )


def test_subscript_dim_check():
    check_single_input("subscript_dim_check", num_subscripts=3)


def test_resize_crop_mirror():
    check_single_input("resize_crop_mirror", crop=[5, 5], resize_shorter=10)


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def get_property_pipeline(files):
    data, _ = fn.readers.file(files=files)
    out = fn.get_property(data, key="source_info")

    return out


def test_get_property():
    root_path = os.path.join(data_root, "db", "single", "png", "0")
    files = [os.path.join(root_path, i) for i in os.listdir(root_path)]

    pipe = get_property_pipeline(files)
    compare_eager_with_pipeline(
        pipe,
        eager.get_property,
        eager_source=PipelineInput(file_reader_pipeline, files=files),
        key="source_info",
    )


def test_tensor_subscript():
    check_single_input("tensor_subscript", lo_0=1, hi_1=-1, at_2=1)


def eager_arithm_ops(data):
    return (
        data * 2,
        data + 2,
        data - 2,
        data / 2,
        data // 2,
        data**2,
        data == 2,
        data != 2,
        data < 2,
        data <= 2,
        data > 2,
        data >= 2,
        data & 2,
        data | 2,
        data ^ 2,
        eager.math.abs(data),
        eager.math.fabs(data),
        eager.math.floor(data),
        eager.math.ceil(data),
        eager.math.pow(data, 2),
        eager.math.fpow(data, 1.5),
        eager.math.min(data, 2),
        eager.math.max(data, 50),
        eager.math.clamp(data, 10, 50),
        eager.math.sqrt(data),
        eager.math.rsqrt(data),
        eager.math.cbrt(data),
        eager.math.exp(data),
        eager.math.exp(data),
        eager.math.log(data),
        eager.math.log2(data),
        eager.math.log10(data),
        eager.math.sin(data),
        eager.math.cos(data),
        eager.math.tan(data),
        eager.math.asin(data),
        eager.math.acos(data),
        eager.math.atan(data),
        eager.math.atan2(data, 3),
        eager.math.sinh(data),
        eager.math.cosh(data),
        eager.math.tanh(data),
        eager.math.asinh(data),
        eager.math.acosh(data),
        eager.math.atanh(data),
    )


def test_arithm_ops():
    with eager.arithmetic():
        pipe = pipeline_arithm_ops_cpu(
            get_data, batch_size=batch_size, num_threads=4, device_id=None
        )
        compare_eager_with_pipeline(pipe, eager_op=eager_arithm_ops)


def test_noise_gaussian():
    check_single_input_stateful("noise.gaussian")


def test_noise_salt_and_pepper():
    check_single_input_stateful("noise.salt_and_pepper")


def test_noise_shot():
    check_single_input_stateful("noise.shot")


def test_random_mask_pixel():
    check_single_input_stateful("segmentation.random_mask_pixel")


def test_random_resized_crop():
    check_single_input_stateful("random_resized_crop", size=[5, 5])


def test_random_object_bbox():
    data = tensors.TensorListCPU(
        [
            tensors.TensorCPU(np.int32([[1, 0, 0, 0], [1, 2, 2, 1], [1, 1, 2, 0], [2, 0, 0, 1]])),
            tensors.TensorCPU(
                np.int32([[0, 3, 3, 0], [1, 0, 1, 2], [0, 1, 1, 0], [0, 2, 0, 1], [0, 2, 2, 1]])
            ),
        ]
    )

    def eager_source(_i, _layout):
        return data

    def fn_source(_):
        return data

    check_single_input_stateful(
        "segmentation.random_object_bbox", fn_source=fn_source, eager_source=eager_source, layout=""
    )


def test_roi_random_crop():
    shape = [10, 20, 3]
    check_single_input_stateful(
        "roi_random_crop",
        crop_shape=[x // 2 for x in shape],
        roi_start=[x // 4 for x in shape],
        roi_shape=[x // 2 for x in shape],
    )


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def random_bbox_crop_pipeline(get_boxes, get_labels, seed):
    boxes = fn.external_source(source=get_boxes)
    labels = fn.external_source(source=get_labels)
    out = fn.random_bbox_crop(
        boxes,
        labels,
        aspect_ratio=[0.5, 2.0],
        thresholds=[0.1, 0.3, 0.5],
        scaling=[0.8, 1.0],
        bbox_layout="xyXY",
        seed=seed,
    )

    return tuple(out)


def test_random_bbox_crop():
    get_boxes = GetData(
        [
            [
                (rng.integers(0, 255, size=[200, 4], dtype=np.uint8) / 255).astype(dtype=np.float32)
                for _ in range(batch_size)
            ]
            for _ in range(data_size)
        ]
    )
    get_labels = GetData(
        [
            [rng.integers(0, 255, size=[200, 1], dtype=np.int32) for _ in range(batch_size)]
            for _ in range(data_size)
        ]
    )

    def eager_source(i, _):
        return get_boxes.eager_source(i), get_labels.eager_source(i)

    _, eager_op, fn_seed = prep_stateful_operators("random_bbox_crop")

    pipe = random_bbox_crop_pipeline(get_boxes.fn_source, get_labels.fn_source, fn_seed)

    compare_eager_with_pipeline(
        pipe,
        eager_op,
        eager_source=eager_source,
        aspect_ratio=[0.5, 2.0],
        thresholds=[0.1, 0.3, 0.5],
        scaling=[0.8, 1.0],
        bbox_layout="xyXY",
    )


def test_random_choice_cpu():
    shape_batch_list = [[np.array(i + 3) for i in range(batch_size)] for _ in range(data_size)]

    data = GetData(shape_batch_list)

    check_single_input_stateful(
        "random.choice",
        fn_source=data.fn_source,
        eager_source=data.eager_source,
        layout=None,
    )


def test_random_coin_flip():
    check_no_input_stateful("random.coin_flip")


def test_normal_distribution():
    check_no_input_stateful("random.normal", shape=[5, 5])


def test_random_uniform():
    check_no_input_stateful("random.uniform")


def test_random_beta():
    check_no_input_stateful("random.beta")


def test_batch_permutation():
    check_no_input_stateful("batch_permutation")


def test_random_crop_generator_cpu():
    shape_batch_list = [
        [np.random.randint(100, 800, size=(2,), dtype=np.int64) for _ in range(batch_size)]
        for _ in range(data_size)
    ]

    data = GetData(shape_batch_list)

    check_single_input_stateful(
        "random_crop_generator",
        fn_source=data.fn_source,
        eager_source=data.eager_source,
        layout=None,
    )


def test_video_decoder():
    filename = os.path.join(get_dali_extra_path(), "db", "video", "cfr", "test_1.mp4")
    data = [
        [np.fromfile(filename, dtype=np.uint8) for _ in range(batch_size)] for _ in range(data_size)
    ]

    data = GetData(data)

    check_single_input(
        "experimental.decoders.video",
        fn_source=data.fn_source,
        eager_source=data.eager_source,
        layout=None,
    )


def test_zeros():
    check_no_input("zeros", shape=(2, 3))


def test_zeros_like():
    check_single_input("zeros_like", layout=None)


def test_ones():
    check_no_input("ones", shape=(2, 3))


def test_ones_like():
    check_single_input("ones_like", layout=None)


def test_full():
    check_single_input("full", shape=(1,))


def test_full_like():
    fill_value = np.array([1, 2, 3], dtype=np.int32)
    array_like = np.zeros((2, 3))

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
    def full_like_pipe():
        return fn.full_like(array_like, fill_value)

    compare_eager_with_pipeline(
        full_like_pipe(),
        lambda x: eager.full_like(x, get_tl([fill_value] * batch_size, None)),
        eager_source=lambda _i, _layout: get_tl([array_like] * batch_size, None),
        layout=None,
    )


def test_io_file_read():
    filenames = [
        os.path.join(get_dali_extra_path(), "db/single/png/0/cat-1046544_640.png"),
    ]

    data = []
    i = 0
    for _ in range(data_size):
        batch = []
        for _ in range(batch_size):
            batch.append(np.frombuffer(filenames[i % len(filenames)].encode(), dtype=np.int8))
            i += 1
        data.append(batch)
    data = GetData(data)

    check_single_input(
        "io.file.read",
        fn_source=data.fn_source,
        eager_source=data.eager_source,
        layout=None,
    )


tested_methods = [
    "decoders.image",
    "decoders.image_crop",
    "decoders.image_slice",
    "decoders.image_random_crop",
    "decoders.numpy",
    "experimental.decoders.image",
    "experimental.decoders.image_crop",
    "experimental.decoders.image_slice",
    "experimental.decoders.image_random_crop",
    "paste",
    "rotate",
    "brightness_contrast",
    "hue",
    "brightness",
    "contrast",
    "hsv",
    "color_twist",
    "saturation",
    "shapes",
    "crop",
    "color_space_conversion",
    "cast",
    "cast_like",
    "resize",
    "experimental.tensor_resize",
    "per_frame",
    "gaussian_blur",
    "laplacian",
    "crop_mirror_normalize",
    "flip",
    "jpeg_compression_distortion",
    "reshape",
    "reinterpret",
    "water",
    "sphere",
    "erase",
    "expand_dims",
    "coord_transform",
    "grid_mask",
    "multi_paste",
    "nonsilent_region",
    "preemphasis_filter",
    "power_spectrum",
    "spectrogram",
    "mel_filter_bank",
    "to_decibels",
    "audio_resample",
    "mfcc",
    "one_hot",
    "transpose",
    "decoders.audio",
    "coord_flip",
    "bb_flip",
    "warp_affine",
    "normalize",
    "lookup_table",
    "slice",
    "pad",
    "readers.file",
    "readers.mxnet",
    "readers.webdataset",
    "readers.coco",
    "readers.caffe",
    "readers.caffe2",
    "readers.nemo_asr",
    "experimental.readers.video",
    "copy",
    "element_extract",
    "bbox_paste",
    "sequence_rearrange",
    "box_encoder",
    "readers.numpy",
    "constant",
    "dump_image",
    "readers.sequence",
    "transforms.translation",
    "transforms.scale",
    "transforms.rotation",
    "transforms.shear",
    "transforms.crop",
    "transforms.combine",
    "reductions.min",
    "reductions.max",
    "reductions.sum",
    "segmentation.select_masks",
    "reductions.mean",
    "reductions.mean_square",
    "reductions.rms",
    "reductions.std_dev",
    "reductions.variance",
    "cat",
    "stack",
    "permute_batch",
    "squeeze",
    "peek_image_shape",
    "experimental.peek_image_shape",
    "subscript_dim_check",
    "get_property",
    "tensor_subscript",
    "arithmetic_generic_op",
    "noise.gaussian",
    "noise.salt_and_pepper",
    "noise.shot",
    "segmentation.random_mask_pixel",
    "segmentation.random_object_bbox",
    "roi_random_crop",
    "random_bbox_crop",
    "random_resized_crop",
    "resize_crop_mirror",
    "random.choice",
    "random.coin_flip",
    "random.normal",
    "random.uniform",
    "random.beta",
    "batch_permutation",
    "random_crop_generator",
    "experimental.decoders.video",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "io.file.read",
    "experimental.warp_perspective",
]

excluded_methods = [
    "hidden.*",
    "jitter",  # not supported for CPU
    "video_reader",  # not supported for CPU
    "video_reader_resize",  # not supported for CPU
    "readers.video",  # not supported for CPU
    "readers.video_resize",  # not supported for CPU
    "optical_flow",  # not supported for CPU
    "experimental.debayer",  # not supported for CPU
    "experimental.equalize",  # not supported for CPU
    "experimental.filter",  # not supported for CPU
    "experimental.inflate",  # not supported for CPU
    "experimental.remap",  # operator is GPU-only
    "experimental.readers.fits",  # lacking test files in DALI_EXTRA
    "experimental.median_blur",  # not supported for CPU
    "experimental.dilate",  # not supported for CPU
    "experimental.erode",  # not supported for CPU
    "experimental.resize",  # not supported for CPU
    "plugin.video.decoder",  # not supported for CPU
]


def test_coverage():
    """Checks coverage of eager operators (almost every operator is also exposed in eager mode).
    If you added a new operator, you should also add a test for it here and add the operator name
    to the ``tested_methods`` list. You should also add eager classification for your operator in
    `dali/python/nvidia/dali/_utils/eager_utils.py`.
    """

    methods = module_functions(eager, remove_prefix="nvidia.dali.experimental.eager")
    methods += module_functions(eager.rng_state(), remove_prefix="rng_state", check_non_module=True)
    # TODO(ksztenderski): Add coverage for GPU operators.
    exclude = "|".join(
        [
            "(^" + x.replace(".", "\.").replace("*", ".*").replace("?", ".") + "$)"  # noqa: W605
            for x in excluded_methods
        ]
    )
    exclude = re.compile(exclude)
    methods = [x for x in methods if not exclude.match(x)]

    assert set(methods).difference(set(tested_methods)) == set(), "Test doesn't cover:\n {}".format(
        set(methods) - set(tested_methods)
    )
