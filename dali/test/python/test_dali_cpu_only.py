# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.math as dmath
import nvidia.dali.plugin.pytorch as pytorch
import torch.utils.dlpack as torch_dlpack
from nvidia.dali.plugin.numba.fn.experimental import numba_function
from test_utils import get_dali_extra_path, get_files, check_batch, RandomlyShapedDataIterator, dali_type, module_functions
from segmentation_test_utils import make_batch_select_masks
from test_audio_decoder_utils import generate_waveforms
import scipy.io.wavfile
from PIL import Image, ImageEnhance
from test_detection_pipeline import coco_anchors
import re

import numpy as np
from nose.tools import assert_raises
import os
import glob
from math import ceil, sqrt
import tempfile
import sys
import json
from collections.abc import Iterable

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
audio_files = get_files(os.path.join('db', 'audio', 'wav'), 'wav')
caffe_dir = os.path.join(data_root, 'db', 'lmdb')
caffe2_dir = os.path.join(data_root, 'db', 'c2lmdb')
recordio_dir = os.path.join(data_root, 'db', 'recordio')
tfrecord_dir = os.path.join(data_root, 'db', 'tfrecord')
coco_dir = os.path.join(data_root, 'db', 'coco', 'images')
coco_annotation = os.path.join(data_root, 'db', 'coco', 'instances.json')
sequence_dir = os.path.join(data_root, 'db', 'sequence', 'frames')

batch_size = 2
test_data_shape = [10, 20, 3]
def get_data():
    out = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
    return out

def test_move_to_device_end():
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    outs = fn.external_source(source=get_data)
    pipe.set_outputs(outs.gpu())
    assert_raises(RuntimeError, pipe.build)

def test_move_to_device_middle():
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source=get_data)
    outs = fn.rotate(data.gpu(), angle=25)
    pipe.set_outputs(outs)
    assert_raises(RuntimeError, pipe.build)

def check_bad_device(device_id):
    test_data_shape = [1, 3, 0, 4]
    def get_data():
        out = [np.empty(test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=device_id)
    outs = fn.external_source(source=get_data, device="gpu")
    pipe.set_outputs(outs)
    assert_raises(RuntimeError, pipe.build)

def test_gpu_op_bad_device():
    for device_id in [None, 0]:
        yield check_bad_device, device_id

def check_mixed_op_bad_device(device_id):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=device_id)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image(input, device="mixed", output_type=types.RGB)
    pipe.set_outputs(decoded)
    assert_raises(RuntimeError, pipe.build)

def test_mixed_op_bad_device():
    for device_id in [None, 0]:
        yield check_bad_device, device_id

def test_image_decoder_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    with pipe:
        input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
        decoded = fn.decoders.image(input, output_type=types.RGB)
        pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def check_single_input(op, input_layout="HWC", get_data=get_data, batch=True, cycle=None, exec_async=True, exec_pipelined=True, **kwargs):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, exec_async=exec_async, exec_pipelined=exec_pipelined)
    with pipe:
        data = fn.external_source(source=get_data, layout=input_layout, batch=batch, cycle=cycle)
        processed = op(data, **kwargs)
        if isinstance(processed, Iterable):
            pipe.set_outputs(*processed)
        else:
            pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def check_no_input(op, get_data=get_data, **kwargs):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    with pipe:
        processed = op(**kwargs)
        if isinstance(processed, Iterable):
            pipe.set_outputs(*processed)
        else:
            pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_rotate_cpu():
    check_single_input(fn.rotate, angle=25)

def test_brightness_contrast_cpu():
    check_single_input(fn.brightness_contrast)

def test_hue_cpu():
    check_single_input(fn.hue)

def test_brightness_cpu():
    check_single_input(fn.brightness)

def test_contrast_cpu():
    check_single_input(fn.contrast)

def test_hsv_cpu():
    check_single_input(fn.hsv)

def test_color_twist_cpu():
    check_single_input(fn.color_twist)

def test_saturation_cpu():
    check_single_input(fn.saturation)

def test_old_color_twist_cpu():
    check_single_input(fn.old_color_twist)

def test_shapes_cpu():
    check_single_input(fn.shapes)

def test_crop_cpu():
    check_single_input(fn.crop, crop=(5, 5))

def test_color_space_coversion_cpu():
    check_single_input(fn.color_space_conversion, image_type=types.BGR, output_type=types.RGB)

def test_cast_cpu():
    check_single_input(fn.cast, dtype=types.INT32)

def test_resize_cpu():
    check_single_input(fn.resize, resize_x=50, resize_y=50)

def test_gaussian_blur_cpu():
    check_single_input(fn.gaussian_blur, window_size=5)

def test_crop_mirror_normalize_cpu():
    check_single_input(fn.crop_mirror_normalize)

def test_flip_cpu():
    check_single_input(fn.flip, horizontal=True)

def test_jpeg_compression_distortion_cpu():
    check_single_input(fn.jpeg_compression_distortion, quality=10)

def test_noise_gaussian_cpu():
    check_single_input(fn.noise.gaussian)

def test_noise_shot_cpu():
    check_single_input(fn.noise.shot)

def test_noise_salt_and_pepper_cpu():
    check_single_input(fn.noise.salt_and_pepper)

def test_image_decoder_crop_device():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image_crop(input, output_type=types.RGB, crop=(10, 10))
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_image_decoder_random_crop_device():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    decoded = fn.decoders.image_random_crop(input, output_type=types.RGB)
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coin_flip_device():
    check_no_input(fn.random.coin_flip)

def test_uniform_device():
    check_no_input(fn.random.uniform)

def test_reshape_cpu():
    new_shape = test_data_shape.copy()
    new_shape[0] //= 2
    new_shape[1] *= 2
    check_single_input(fn.reshape, shape=new_shape)

def test_reinterpret_cpu():
    check_single_input(fn.reinterpret, rel_shape=[0.5, 1, -1])

def test_water_cpu():
    check_single_input(fn.water)

def test_sphere_cpu():
    check_single_input(fn.sphere)

def test_erase_cpu():
    check_single_input(fn.erase, anchor=[0.3], axis_names="H", normalized_anchor=True, shape=[0.1],normalized_shape=True)

def test_random_resized_crop_cpu():
    check_single_input(fn.random_resized_crop, size=[5, 5])

def test_expand_dims_cpu():
    check_single_input(fn.expand_dims, axes=1, new_axis_names="Z")

def test_coord_transform_cpu():
    M= [0, 0, 1,
        0, 1, 0,
        1, 0, 0]
    check_single_input(fn.coord_transform, M=M, dtype=types.UINT8)

def test_grid_mask_cpu():
    check_single_input(fn.grid_mask, tile=51, ratio=0.38158387, angle=2.6810782)

def test_multi_paste_cpu():
    check_single_input(fn.multi_paste, in_ids=np.array([0, 1]), output_size=test_data_shape)

def test_roi_random_crop_cpu():
    check_single_input(fn.roi_random_crop, crop_shape=[x // 2 for x in test_data_shape], roi_start=[x // 4 for x in test_data_shape], roi_shape=[x // 2 for x in test_data_shape])

def test_random_object_bbox_cpu():
    get_data = [
        np.int32([[1, 0, 0, 0],
                [1, 2, 2, 1],
                [1, 1, 2, 0],
                [2, 0, 0, 1]]),

        np.int32([[0, 3, 3, 0],
                [1, 0, 1, 2],
                [0, 1, 1, 0],
                [0, 2, 0, 1],
                [0, 2, 2, 1]])
    ]
    check_single_input(fn.segmentation.random_object_bbox, get_data=get_data, batch=False, cycle="quiet", input_layout="")

def test_numba_func_cpu():
    def set_all_values_to_255_batch(out0, in0):
        out0[0][:] = 255

    def setup_out_shape(out_shape, in_shape):
        out_shape = in_shape

    check_single_input(numba_function,
                       run_fn=set_all_values_to_255_batch, out_types=[types.UINT8], in_types=[types.UINT8],
                       outs_ndim=[3], ins_ndim=[3], setup_fn=setup_out_shape, batch_processing=True)

def test_dl_tensor_python_function_cpu():
    def dl_tensor_operation(tensor):
        tensor = torch_dlpack.from_dlpack(tensor)
        tensor_n = tensor.double() / 255
        ret = tensor_n.sin()
        ret = torch_dlpack.to_dlpack(ret)
        return ret

    def batch_dl_tensor_operation(tensors):
        out = [dl_tensor_operation(t) for t in tensors]
        return out

    check_single_input(fn.dl_tensor_python_function, function=batch_dl_tensor_operation, batch_processing=True, exec_async=False, exec_pipelined=False)

def test_nonsilent_region_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200]
    def get_data():
        out = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        out[0][0] = 0
        out[1][0] = 0
        out[1][1] = 0
        return out
    data = fn.external_source(source=get_data)
    processed, _ = fn.nonsilent_region(data)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

test_audio_data_shape = [200]
def get_audio_data():
    out = [np.random.ranf(size=test_audio_data_shape).astype(dtype=np.float32) for _ in range(batch_size)]
    return out

def test_preemphasis_filter_cpu():
    check_single_input(fn.preemphasis_filter, get_data=get_audio_data, input_layout=None)

def test_power_spectrum_cpu():
    check_single_input(fn.power_spectrum, get_data=get_audio_data, input_layout=None)

def test_spectrogram_cpu():
    check_single_input(fn.spectrogram, get_data=get_audio_data, input_layout=None, nfft=60, window_length=50, window_step=25)

def test_mel_filter_bank_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_audio_data)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    processed = fn.mel_filter_bank(spectrum)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_to_decibels_cpu():
    check_single_input(fn.to_decibels, get_data=get_audio_data, input_layout=None)

def test_mfcc_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_audio_data)
    spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
    mel = fn.mel_filter_bank(spectrum)
    dec = fn.to_decibels(mel)
    processed = fn.mfcc(dec)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_fast_resize_crop_mirror_cpu():
    check_single_input(fn.fast_resize_crop_mirror, crop=[5, 5], resize_shorter=10)

def test_resize_crop_mirror_cpu():
    check_single_input(fn.resize_crop_mirror, crop=[5, 5], resize_shorter=10)

def test_normal_distribution_cpu():
    check_no_input(fn.random.normal, shape=[5, 5])

def test_one_hot_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200]
    def get_data():
        out = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source=get_data)
    processed = fn.one_hot(data, num_classes=256)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_transpose_cpu():
    check_single_input(fn.transpose, perm=[2, 0, 1])

def test_audio_decoder_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(files=audio_files, shard_id=0, num_shards=1)
    decoded, _ = fn.decoders.audio(input)
    pipe.set_outputs(decoded)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coord_flip_cpu():
    test_data_shape = [200, 2]
    def get_data():
        out = [(np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    check_single_input(fn.coord_flip, get_data=get_data, input_layout=None)

def test_bb_flip_cpu():
    test_data_shape = [200, 4]
    def get_data():
        out = [(np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    check_single_input(fn.bb_flip, get_data=get_data, input_layout=None)

def test_warp_affine_cpu():
    warp_matrix = (0.1, 0.9, 10, 0.8, -0.2, -20)
    check_single_input(fn.warp_affine, matrix=warp_matrix)

def test_normalize_cpu():
    check_single_input(fn.normalize, batch=True)

def test_lookup_table_cpu():
    test_data_shape = [100]
    def get_data():
        out = [np.random.randint(0, 5, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out
    check_single_input(fn.lookup_table, keys=[1, 3], values=[10, 50], get_data=get_data, input_layout=None)

def test_slice_cpu():
    anch_shape = [2]
    def get_anchors():
        out = [(np.random.randint(1, 256, size=anch_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    def get_shape():
        out = [(np.random.randint(1, 256, size=anch_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    anchors = fn.external_source(source=get_anchors)
    shape = fn.external_source(source=get_shape)
    processed = fn.slice(data, anchors, shape, out_of_bounds_policy="pad")
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_image_decoder_slice_cpu():
    anch_shape = [2]
    def get_anchors():
        out = [(np.random.randint(1, 128, size=anch_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    def get_shape():
        out = [(np.random.randint(1, 128, size=anch_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    anchors = fn.external_source(source=get_anchors)
    shape = fn.external_source(source=get_shape)
    processed = fn.decoders.image_slice(input, anchors, shape)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_pad_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [5, 4, 3]
    def get_data():
        out = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out
    data = fn.external_source(source=get_data, layout="HWC")
    processed = fn.pad(data, fill_value=-1, axes=(0,), shape=(10,) )
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_mxnet_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    out, _ = fn.readers.mxnet(path=os.path.join(recordio_dir, "train.rec"),
                              index_path=os.path.join(recordio_dir, "train.idx"),
                              shard_id=0, num_shards=1)
    check_no_input(fn.readers.mxnet, path=os.path.join(recordio_dir, "train.rec"),
                   index_path=os.path.join(recordio_dir, "train.idx"),
                   shard_id=0, num_shards=1)

def test_tfrecord_reader_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    tfrecord = sorted(glob.glob(os.path.join(tfrecord_dir, '*[!i][!d][!x]')))
    tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_dir, '*idx')))
    input = fn.readers.tfrecord(path=tfrecord,
                                index_path=tfrecord_idx,
                                shard_id=0, num_shards=1,
                                features={"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                          "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)})
    out = input["image/encoded"]
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_coco_reader_cpu():
    check_no_input(fn.readers.coco, file_root=coco_dir, annotations_file=coco_annotation, shard_id=0, num_shards=1)

def test_caffe_reader_cpu():
    check_no_input(fn.readers.caffe, path=caffe_dir, shard_id=0, num_shards=1)

def test_caffe2_reader_cpu():
    check_no_input(fn.readers.caffe2, path=caffe2_dir, shard_id=0, num_shards=1)

def test_nemo_asr_reader_cpu():
    with tempfile.TemporaryDirectory() as tmp_dir:
        def create_manifest_file(manifest_file, names, lengths, rates, texts):
            assert(len(names) == len(lengths) == len(rates) == len(texts))
            data = []
            for idx in range(len(names)):
                entry_i = {}
                entry_i['audio_filepath'] = names[idx]
                entry_i['duration'] = lengths[idx] * (1.0 / rates[idx])
                entry_i["text"] = texts[idx]
                data.append(entry_i)
            with open(manifest_file, 'w') as f:
                for entry in data:
                    json.dump(entry, f)
                    f.write('\n')
        nemo_asr_manifest = os.path.join(tmp_dir, "nemo_asr_manifest.json")
        names = [
            os.path.join(tmp_dir, "dali_test_1C.wav"),
            os.path.join(tmp_dir, "dali_test_2C.wav"),
            os.path.join(tmp_dir, "dali_test_4C.wav")
        ]

        freqs = [
            np.array([0.02]),
            np.array([0.01, 0.012]),
            np.array([0.01, 0.012, 0.013, 0.014])
        ]
        rates = [ 22050, 22050, 12347 ]
        lengths = [ 10000, 54321, 12345 ]

        def create_ref():
            ref = []
            for i in range(len(names)):
                wave = generate_waveforms(lengths[i], freqs[i])
                wave = (wave * 32767).round().astype(np.int16)
                ref.append(wave)
            return ref

        ref_i = create_ref()

        def create_wav_files():
            for i in range(len(names)):
                scipy.io.wavfile.write(names[i], rates[i], ref_i[i])

        create_wav_files()

        ref_text_literal = [
            "dali test 1C",
            "dali test 2C",
            "dali test 4C",
        ]
        nemo_asr_manifest = os.path.join(tmp_dir, "nemo_asr_manifest.json")
        create_manifest_file(nemo_asr_manifest, names, lengths, rates, ref_text_literal)

        fixed_seed = 1234
        check_no_input(fn.readers.nemo_asr, manifest_filepaths=[nemo_asr_manifest], dtype=types.INT16, downmix=False,
                                            read_sample_rate=True, read_text=True, seed=fixed_seed)

def test_copy_cpu():
    check_single_input(fn.copy)

def test_element_extract_cpu():
    check_single_input(fn.element_extract, element_map=[0, 3], input_layout=None)

def test_bbox_paste_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_data_shape = [200, 4]
    def get_data():
        out = [(np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source=get_data)
    paste_posx = fn.random.uniform(range=(0, 1))
    paste_posy = fn.random.uniform(range=(0, 1))
    paste_ratio = fn.random.uniform(range=(1, 2))
    processed = fn.bbox_paste(data, paste_x=paste_posx, paste_y=paste_posy, ratio=paste_ratio)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_random_bbox_crop_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [200, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size=test_box_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [200, 1]
    def get_lables():
        out = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return out
    boxes = fn.external_source(source=get_boxes)
    lables = fn.external_source(source=get_lables)
    processed, _, _, _ = fn.random_bbox_crop(boxes, lables,
                                             aspect_ratio=[0.5, 2.0],
                                             thresholds=[0.1, 0.3, 0.5],
                                             scaling=[0.8, 1.0],
                                             bbox_layout="xyXY")
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_ssd_random_crop_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [200, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size=test_box_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [200]
    def get_lables():
        out = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return out
    data = fn.external_source(source=get_data)
    boxes = fn.external_source(source=get_boxes)
    lables = fn.external_source(source=get_lables)
    processed, _, _ = fn.ssd_random_crop(data, boxes, lables)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_sequence_rearrange_cpu():
    test_data_shape = [5, 10, 20, 3]
    def get_data():
        out = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out
    check_single_input(fn.sequence_rearrange, new_order=[0, 4, 1, 3, 2], get_data=get_data, input_layout="FHWC")

def test_box_encoder_cpu():

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    test_box_shape = [20, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size=test_box_shape, dtype=np.uint8) / 255).astype(dtype=np.float32) for _ in range(batch_size)]
        return out
    test_lables_shape = [20, 1]
    def get_lables():
        out = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return out
    boxes = fn.external_source(source=get_boxes)
    lables = fn.external_source(source=get_lables)
    processed, _ = fn.box_encoder(boxes, lables, anchors=coco_anchors())
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_numpy_reader_cpu():
    with tempfile.TemporaryDirectory() as test_data_root:
        rng = np.random.RandomState(12345)
        def create_numpy_file(filename, shape, typ, fortran_order):
            # generate random array
            arr = rng.random_sample(shape) * 10.
            arr = arr.astype(typ)
            if fortran_order:
                arr = np.asfortranarray(arr)
            np.save(filename, arr)

        num_samples = 20
        filenames = []
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)

        check_no_input(fn.readers.numpy, file_root=test_data_root)

def test_python_function_cpu():
    def resize(image):
        return np.array(Image.fromarray(image).resize((50, 10)))

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, exec_async=False, exec_pipelined=False)
    check_single_input(fn.python_function, function=resize, exec_async=False, exec_pipelined=False)

def test_constant_cpu():
    check_no_input(fn.constant, fdata = (1.25,2.5,3))

def test_dump_image_cpu():
    check_single_input(fn.dump_image)

def test_sequence_reader_cpu():
    check_no_input(fn.readers.sequence, file_root=sequence_dir, sequence_length=2, shard_id=0, num_shards=1)

def test_affine_translate_cpu():
    check_no_input(fn.transforms.translation, offset=(2, 3))

def test_affine_scale_cpu():
    check_no_input(fn.transforms.scale, scale=(2, 3))

def test_affine_rotate_cpu():
    check_no_input(fn.transforms.rotation, angle=30.0)

def test_affine_shear_cpu():
    check_no_input(fn.transforms.shear, shear=(2., 1.))

def test_affine_crop_cpu():
    check_no_input(fn.transforms.crop,
        from_start=(0., 1.), from_end=(1., 1.), to_start=(0.2, 0.3), to_end=(0.8, 0.5))

def test_combine_transforms_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    with pipe:
        t = fn.transforms.translation(offset=(1, 2))
        r = fn.transforms.rotation(angle=30.0)
        s = fn.transforms.scale(scale=(2, 3))
        out = fn.transforms.combine(t, r, s)
    pipe.set_outputs(out)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_min_cpu():
    check_single_input(fn.reductions.min)

def test_reduce_max_cpu():
    check_single_input(fn.reductions.max)

def test_reduce_sum_cpu():
    check_single_input(fn.reductions.sum)

def test_segmentation_select_masks():
    def get_data_source(*args, **kwargs):
        return lambda: make_batch_select_masks(*args, **kwargs)
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, seed=1234)
    with pipe:
        polygons, vertices, selected_masks=fn.external_source(
            num_outputs = 3, device='cpu',
            source = get_data_source(batch_size, vertex_ndim=2, npolygons_range=(1, 5),
                                     nvertices_range=(3, 10))
        )
        out_polygons, out_vertices = fn.segmentation.select_masks(
            selected_masks, polygons, vertices, reindex_masks=False
        )
    pipe.set_outputs(polygons, vertices, selected_masks, out_polygons, out_vertices)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_mean_cpu():
    check_single_input(fn.reductions.mean)

def test_reduce_mean_square_cpu():
    check_single_input(fn.reductions.mean_square)

def test_reduce_root_mean_square_cpu():
    check_single_input(fn.reductions.rms)

def test_reduce_std_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_data)
    mean = fn.reductions.mean(data)
    reduced = fn.reductions.std_dev(data, mean)
    pipe.set_outputs(reduced)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_reduce_variance_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_data)
    mean = fn.reductions.mean(data)
    reduced = fn.reductions.variance(data, mean)
    pipe.set_outputs(reduced)

def test_arithm_ops_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    processed = [data * 2,
                 data + 2,
                 data - 2,
                 data / 2,
                 data // 2,
                 data ** 2,
                 data == 2,
                 data != 2,
                 data < 2,
                 data <= 2,
                 data > 2,
                 data >= 2,
                 data & 2,
                 data | 2,
                 data ^ 2,
                 dmath.abs(data),
                 dmath.fabs(data),
                 dmath.floor(data),
                 dmath.ceil(data),
                 dmath.pow(data, 2),
                 dmath.fpow(data, 1.5),
                 dmath.min(data, 2),
                 dmath.max(data, 50),
                 dmath.clamp(data, 10, 50),
                 dmath.sqrt(data),
                 dmath.rsqrt(data),
                 dmath.cbrt(data),
                 dmath.exp(data),
                 dmath.exp(data),
                 dmath.log(data),
                 dmath.log2(data),
                 dmath.log10(data),
                 dmath.sin(data),
                 dmath.cos(data),
                 dmath.tan(data),
                 dmath.asin(data),
                 dmath.acos(data),
                 dmath.atan(data),
                 dmath.atan2(data, 3),
                 dmath.sinh(data),
                 dmath.cosh(data),
                 dmath.tanh(data),
                 dmath.asinh(data),
                 dmath.acosh(data),
                 dmath.atanh(data)]
    pipe.set_outputs(*processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_arithm_ops_cpu_gpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    processed = [data * data.gpu(),
                 data + data.gpu(),
                 data - data.gpu(),
                 data / data.gpu(),
                 data // data.gpu(),
                 data ** data.gpu(),
                 data == data.gpu(),
                 data != data.gpu(),
                 data < data.gpu(),
                 data <= data.gpu(),
                 data > data.gpu(),
                 data >= data.gpu(),
                 data & data.gpu(),
                 data | data.gpu(),
                 data ^ data.gpu()]
    pipe.set_outputs(*processed)
    assert_raises(RuntimeError, pipe.build)

def test_pytorch_plugin_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    outs = fn.external_source(source=get_data, layout="HWC")
    pipe.set_outputs(outs)
    pii = pytorch.DALIGenericIterator([pipe], ["data"])

def test_random_mask_pixel_cpu():
    check_single_input(fn.segmentation.random_mask_pixel)

def test_cat_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    data2 = fn.external_source(source=get_data, layout="HWC")
    data3 = fn.external_source(source=get_data, layout="HWC")
    pixel_pos = fn.cat(data, data2, data3)
    pipe.set_outputs(pixel_pos)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_stack_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    data2 = fn.external_source(source=get_data, layout="HWC")
    data3 = fn.external_source(source=get_data, layout="HWC")
    pixel_pos = fn.stack(data, data2, data3)
    pipe.set_outputs(pixel_pos)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_batch_permute_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
    data = fn.external_source(source=get_data, layout="HWC")
    perm = fn.batch_permutation(seed=420)
    processed = fn.permute_batch(data, indices=perm)
    pipe.set_outputs(processed)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_squeeze_cpu():
    test_data_shape = [10, 20, 3, 1, 1]
    def get_data():
        out = [np.zeros(shape=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return out
    check_single_input(fn.squeeze, axis_names="YZ", get_data=get_data, input_layout="HWCYZ")

def test_peek_image_shape_cpu():
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None)
    input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    shapes = fn.peek_image_shape(input)
    pipe.set_outputs(shapes)
    pipe.build()
    for _ in range(3):
        pipe.run()

def test_separated_exec_setup():
    batch_size = 128
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None, prefetch_queue_depth={"cpu_size": 5, "gpu_size": 3})
    inputs, labels = fn.readers.caffe(path=caffe_dir, shard_id=0, num_shards=1)
    images = fn.decoders.image(inputs, output_type=types.RGB)
    images = fn.resize(images, resize_x=224, resize_y=224)
    images_cpu = fn.dump_image(images, suffix="cpu")
    pipe.set_outputs(images, images_cpu)

    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    a_raw = out[0]
    a_cpu = out[1]
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)

def test_tensor_subscript():
    pipe = Pipeline(batch_size=3, num_threads=3, device_id=None)
    input = fn.external_source(source=get_data)
    pipe.set_outputs(input[1:,:-1,1])
    pipe.build()
    out, = pipe.run()
    assert out.at(0).shape == np.zeros(test_data_shape)[1:,:-1,1].shape

def test_subscript_dim_check():
    check_single_input(fn.subscript_dim_check, num_subscripts=3)

tested_methods = [
    "audio_decoder",
    "image_decoder",
    "image_decoder_slice",
    "image_decoder_crop",
    "image_decoder_random_crop",
    "decoders.image",
    "decoders.image_crop",
    "decoders.image_slice",
    "decoders.image_random_crop",
    "decoders.audio",
    "external_source",
    "stack",
    "reductions.variance",
    "reductions.std_dev",
    "reductions.rms",
    "reductions.mean",
    "reductions.mean_square",
    "reductions.max",
    "reductions.min",
    "reductions.sum",
    "transforms.translation",
    "transforms.rotation",
    "transforms.scale",
    "transforms.combine",
    "transforms.shear",
    "transforms.crop",
    "transform_translation",
    "crop",
    "constant",
    "dump_image",
    "numpy_reader",
    "tfrecord_reader",
    "file_reader",
    "sequence_reader",
    "mxnet_reader",
    "caffe_reader",
    "caffe2_reader",
    "coco_reader",
    "nemo_asr_reader",
    "readers.nemo_asr",
    "readers.file",
    "readers.sequence",
    "readers.tfrecord",
    "readers.mxnet",
    "readers.caffe",
    "readers.caffe2",
    "readers.coco",
    "readers.numpy",
    "coin_flip",
    "uniform",
    "random.uniform",
    "random.coin_flip",
    "random.normal",
    "random_bbox_crop",
    "python_function",
    "rotate",
    "brightness_contrast",
    "hue",
    "brightness",
    "contrast",
    "hsv",
    "color_twist",
    "saturation",
    "old_color_twist",
    "shapes",
    "crop",
    "color_space_conversion",
    "cast",
    "resize",
    "gaussian_blur",
    "crop_mirror_normalize",
    "flip",
    "jpeg_compression_distortion",
    "noise.shot",
    "noise.gaussian",
    "noise.salt_and_pepper",
    "reshape",
    "reinterpret",
    "water",
    "sphere",
    "erase",
    "random_resized_crop",
    "ssd_random_crop",
    "bbox_paste",
    "coord_flip",
    "cat",
    "bb_flip",
    "warp_affine",
    "normalize",
    "pad",
    "preemphasis_filter",
    "power_spectrum",
    "spectrogram",
    "to_decibels",
    "sequence_rearrange",
    "normal_distribution",
    "mel_filter_bank",
    "nonsilent_region",
    "one_hot",
    "copy",
    "resize_crop_mirror",
    "fast_resize_crop_mirror",
    "segmentation.select_masks",
    "slice",
    "segmentation.random_mask_pixel",
    "transpose",
    "paste",
    "mfcc",
    "lookup_table",
    "element_extract",
    "arithmetic_generic_op",
    "box_encoder",
    "permute_batch",
    "batch_permutation",
    "squeeze",
    "peek_image_shape",
    "expand_dims",
    "coord_transform",
    "grid_mask",
    "multi_paste",
    "roi_random_crop",
    "segmentation.random_object_bbox",
    "tensor_subscript",
    "subscript_dim_check",
    "math.ceil",
    "math.clamp",
    "math.tanh",
    "math.tan",
    "math.log2",
    "math.atanh",
    "math.atan",
    "math.atan2",
    "math.sin",
    "math.cos",
    "math.asinh",
    "math.abs",
    "math.sqrt",
    "math.exp",
    "math.acos",
    "math.log",
    "math.fabs",
    "math.sinh",
    "math.rsqrt",
    "math.asin",
    "math.floor",
    "math.cosh",
    "math.log10",
    "math.max",
    "math.cbrt",
    "math.pow",
    "math.fpow",
    "math.acosh",
    "math.min",
    "numba.fn.experimental.numba_function",
    "dl_tensor_python_function",
]

excluded_methods = [
    "hidden.*",
    "jitter",               # not supported for CPU
    "video_reader",         # not supported for CPU
    "video_reader_resize",  # not supported for CPU
    "readers.video",        # not supported for CPU
    "readers.video_resize", # not supported for CPU
    "optical_flow",         # not supported for CPU
]

def test_coverage():
    methods = module_functions(fn, remove_prefix="nvidia.dali.fn")
    methods += module_functions(dmath, remove_prefix = "nvidia.dali")
    exclude = "|".join(["(^" + x.replace(".", "\.").replace("*", ".*").replace("?", ".") + "$)" for x in excluded_methods])
    exclude = re.compile(exclude)
    methods = [x for x in methods if not exclude.match(x)]
    # we are fine with covering more we can easily list, like numba
    assert set(methods).difference(set(tested_methods)) == set(), "Test doesn't cover:\n {}".format(set(methods) - set(tested_methods))
