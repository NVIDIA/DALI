# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from functools import reduce

from nvidia.dali import pipeline_def
import nvidia.dali.experimental.eager as eager
import nvidia.dali.fn as fn
import nvidia.dali.tensors as tensors
import nvidia.dali.types as types
from nvidia.dali._utils.eager_utils import _slice_tensorlist
from test_dali_cpu_only_utils import setup_test_nemo_asr_reader_cpu, setup_test_numpy_reader_cpu
from test_utils import check_batch, get_dali_extra_path, get_files
from webdataset_base import generate_temp_index_file as generate_temp_wds_index

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
caffe_dir = os.path.join(data_root, 'db', 'lmdb')
caffe2_dir = os.path.join(data_root, 'db', 'c2lmdb')
recordio_dir = os.path.join(data_root, 'db', 'recordio')
webdataset_dir = os.path.join(data_root, 'db', 'webdataset')
coco_dir = os.path.join(data_root, 'db', 'coco', 'images')
coco_annotation = os.path.join(data_root, 'db', 'coco', 'instances.json')
sequence_dir = os.path.join(data_root, 'db', 'sequence', 'frames')
video_files = get_files(os.path.join('db', 'video', 'vfr'), 'mp4')

rng = np.random.default_rng()

batch_size = 2
data_size = 10
sample_shape = [20, 20, 3]
data = [[rng.integers(0, 255, size=sample_shape, dtype=np.uint8)
         for _ in range(batch_size)] for _ in range(data_size)]


def get_tl(data, layout='HWC'):
    layout = '' if layout is None or (data.ndim != 4 and layout == 'HWC') else layout
    return tensors.TensorListCPU(data, layout=layout)


def get_data(i):
    return data[i]


def get_data_eager(i, layout='HWC'):
    return get_tl(np.array(get_data(i)), layout)


def get_ops(op_path, fn_op=None, eager_op=None):
    import_path = op_path.split('.')
    if fn_op is None:
        fn_op = reduce(getattr, [fn] + import_path)
    if eager_op is None:
        eager_op = reduce(getattr, [eager] + import_path)
    return fn_op, eager_op


def compare_eager_with_pipeline(pipe, eager_op, *, eager_source=get_data_eager, layout='HWC',
                                batch_size=batch_size, N_iterations=5, **kwargs):
    pipe.build()
    for i in range(N_iterations):
        input_tl = eager_source(i, layout)
        out_fn = pipe.run()
        if isinstance(input_tl, (tuple, list)):
            out_eager = eager_op(*input_tl, **kwargs)
        else:
            out_eager = eager_op(input_tl, **kwargs)

        if not isinstance(out_eager, (tuple, list)):
            out_eager = (out_eager,)

        for o_fn, o_eager in zip(out_fn, out_eager):
            assert type(o_fn) == type(o_eager)
            check_batch(o_fn, o_eager, batch_size)


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def single_op_pipeline(op, kwargs, source=get_data, layout='HWC'):
    data = fn.external_source(source=source, layout=layout)
    out = op(data, **kwargs)

    if isinstance(out, list):
        out = tuple(out)
    return out


def check_single_input(op_path, *, pipe_fun=single_op_pipeline, fn_source=get_data, fn_op=None,
                       eager_source=get_data_eager, eager_op=None, layout='HWC', **kwargs):

    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = pipe_fun(fn_op, kwargs, source=fn_source, layout=layout)

    compare_eager_with_pipeline(pipe, eager_op, eager_source=eager_source, layout=layout, **kwargs)


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reader_pipeline(op, kwargs):
    out = op(pad_last_batch=True, **kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def check_reader(op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=2, **kwargs):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = reader_pipeline(fn_op, kwargs)
    pipe.build()

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

                assert type(tensor_out_fn) == type(tensor_out_eager)
                check_batch(tensor_out_fn, tensor_out_eager, len(tensor_out_eager))


def test_rotate_cpu():
    check_single_input('rotate', angle=25)


def test_brightness_contrast_cpu():
    check_single_input('brightness_contrast')


def test_hue_cpu():
    check_single_input('hue')


def test_brightness_cpu():
    check_single_input('brightness')


def test_contrast_cpu():
    check_single_input('contrast')


def test_hsv_cpu():
    check_single_input('hsv')


def test_color_twist_cpu():
    check_single_input('color_twist')


def test_saturation_cpu():
    check_single_input('saturation')


def test_shapes_cpu():
    check_single_input('shapes')


def test_crop_cpu():
    check_single_input('crop', crop=(5, 5))


def test_color_space_coversion_cpu():
    check_single_input('color_space_conversion', image_type=types.BGR, output_type=types.RGB)


def test_cast_cpu():
    check_single_input('cast', dtype=types.INT32)


def test_resize_cpu():
    check_single_input('resize', resize_x=50, resize_y=50)


def test_per_frame_cpu():
    check_single_input('per_frame', replace=True)


def test_gaussian_blur_cpu():
    check_single_input('gaussian_blur', window_size=5)


def test_laplacian_cpu():
    check_single_input('laplacian', window_size=5)


def test_crop_mirror_normalize_cpu():
    check_single_input('crop_mirror_normalize')


def test_flip_cpu():
    check_single_input('flip', horizontal=True)


def test_jpeg_compression_distortion_cpu():
    check_single_input('jpeg_compression_distortion', quality=10)


def test_reshape_cpu():
    new_shape = sample_shape.copy()
    new_shape[0] //= 2
    new_shape[1] *= 2
    check_single_input('reshape', shape=new_shape)


def test_reinterpret_cpu():
    check_single_input('reinterpret', rel_shape=[0.5, 1, -1])


def test_water_cpu():
    check_single_input('water')


def test_sphere_cpu():
    check_single_input('sphere')


def test_erase_cpu():
    check_single_input('erase', anchor=[0.3], axis_names='H',
                       normalized_anchor=True, shape=[0.1], normalized_shape=True)


def test_expand_dims_cpu():
    check_single_input('expand_dims', axes=1, new_axis_names='Z')


def test_coord_transform_cpu():
    M = [0, 0, 1,
         0, 1, 0,
         1, 0, 0]
    check_single_input('coord_transform', M=M, dtype=types.UINT8)


def test_grid_mask_cpu():
    check_single_input('grid_mask', tile=51, ratio=0.38158387, angle=2.6810782)


def test_multi_paste_cpu():
    check_single_input('multi_paste', in_ids=np.array([0, 1]), output_size=sample_shape)


def test_file_reader_cpu():
    check_reader('readers.file', file_root=images_dir)


def test_mxnet_reader_cpu():
    check_reader('readers.mxnet', path=os.path.join(recordio_dir, 'train.rec'),
                 index_path=os.path.join(recordio_dir, 'train.idx'), shard_id=0, num_shards=1)


def test_webdataset_reader_cpu():
    webdataset = os.path.join(webdataset_dir, 'MNIST', 'devel-0.tar')
    webdataset_idx = generate_temp_wds_index(webdataset)
    check_reader('readers.webdataset',
                 paths=webdataset,
                 index_paths=webdataset_idx.name,
                 ext=['jpg', 'cls'],
                 shard_id=0, num_shards=1)


def test_coco_reader_cpu():
    check_reader('readers.coco', file_root=coco_dir,
                 annotations_file=coco_annotation, shard_id=0, num_shards=1)


def test_caffe_reader_cpu():
    check_reader('readers.caffe', path=caffe_dir, shard_id=0, num_shards=1)


def test_caffe2_reader_cpu():
    check_reader('readers.caffe2', path=caffe2_dir, shard_id=0, num_shards=1)


def test_nemo_asr_reader_cpu():
    tmp_dir, nemo_asr_manifest = setup_test_nemo_asr_reader_cpu()

    with tmp_dir:
        check_reader('readers.nemo_asr', manifest_filepaths=[nemo_asr_manifest], dtype=types.INT16,
                     downmix=False, read_sample_rate=True, read_text=True, seed=1234)


def test_video_reader():
    check_reader('experimental.readers.video', filenames=video_files,
                 labels=[0, 1], sequence_length=10)


def test_numpy_reader_cpu():
    with setup_test_numpy_reader_cpu() as test_data_root:
        check_reader('readers.numpy', file_root=test_data_root)


def test_sequence_reader_cpu():
    check_reader('readers.sequence', file_root=sequence_dir,
                 sequence_length=2, shard_id=0, num_shards=1)
