# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import ctypes
import os
from dataclasses import dataclass
import paddle
import numpy as np
from nvidia.dali.backend import TensorListCPU
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.paddle import DALIGenericIterator
from utils.mode import Mode
from utils.utility import get_num_trainers, get_trainer_id


@dataclass
class PipeOpMeta:
    crop: int
    resize_shorter: int
    min_area: float
    max_area: float
    lower: float
    upper: float
    interp: types.DALIInterpType
    mean: float
    std: float
    output_dtype: types.DALIDataType
    output_layout: str
    pad_output: bool

@pipeline_def(seed=12)
def create_dali_pipeline(data_dir, ops_meta, shard_id, num_shards, dali_cpu=False,
                         is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask HW nvJPEG to allocate memory ahead for the biggest image in the data set to
    # avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                                device=decoder_device,
                                                output_type=types.RGB,
                                                preallocate_width_hint=preallocate_width_hint,
                                                preallocate_height_hint=preallocate_height_hint,
                                                random_aspect_ratio=[
                                                    ops_meta.lower, ops_meta.upper],
                                                random_area=[ops_meta.min_area,
                                                            ops_meta.max_area],
                                                num_attempts=100)
        images = fn.resize(images,
                            resize_x=ops_meta.crop,
                            resize_y=ops_meta.crop,
                            interp_type=ops_meta.interp)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                    device=decoder_device,
                                    output_type=types.RGB)
        images = fn.resize(images,
                            resize_shorter=ops_meta.resize_shorter,
        interp_type=ops_meta.interp)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                        dtype=ops_meta.output_dtype,
                                        output_layout=ops_meta.output_layout,
                                        crop=(ops_meta.crop, ops_meta.crop),
                                        mean=ops_meta.mean,
                                        std=ops_meta.std,
                                        pad_output=ops_meta.pad_output,
                                        mirror=mirror)
    labels = labels.gpu()
    labels = fn.cast(labels, dtype=types.INT64)
    return images, labels


def dali_dataloader(args, mode, device):
    """
    Define a dali dataloader with configuration to operate datasets.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        mode(utils.Mode): Train or eval mode.
        device(int): Id of GPU to load data.
    Outputs:
        DALIGenericIterator(nvidia.dali.plugin.paddle.DALIGenericIterator)
            Iteratable outputs of DALI pipeline,
            including "data" and "label" in type of Paddle's Tensor.
    """
    assert "gpu" in device, "gpu training is required for DALI"
    assert mode in Mode, "Dataset mode should be in supported Modes"

    device_id = int(device.split(':')[1])

    seed = args.dali_random_seed
    num_threads = args.dali_num_threads
    batch_size = args.batch_size

    interp = 1  # settings.interpolation or 1  # default to linear
    interp_map = {
        # cv2.INTER_NEAREST
        0: types.DALIInterpType.INTERP_NN,
        # cv2.INTER_LINEAR
        1: types.DALIInterpType.INTERP_LINEAR,
        # cv2.INTER_CUBIC
        2: types.DALIInterpType.INTERP_CUBIC,
        # LANCZOS3 for cv2.INTER_LANCZOS4
        3: types.DALIInterpType.INTERP_LANCZOS3
    }
    assert interp in interp_map, "interpolation method not supported by DALI"
    interp = interp_map[interp]

    normalize_scale = args.normalize_scale
    normalize_mean = args.normalize_mean
    normalize_std = args.normalize_std
    normalize_mean = [v / normalize_scale for v in normalize_mean]
    normalize_std = [v / normalize_scale for v in normalize_std]

    output_layout = args.data_layout[1:]  # NCHW -> CHW or NHWC -> HWC
    pad_output = args.image_channel == 4
    output_dtype = types.FLOAT16 if args.dali_output_fp16 else types.FLOAT

    shard_id = get_trainer_id()
    num_shards = get_num_trainers()

    scale = args.rand_crop_scale
    ratio = args.rand_crop_ratio

    ops_meta = PipeOpMeta(
        crop=args.crop_size,
        resize_shorter=args.resize_short,
        min_area=scale[0],
        max_area=scale[1],
        lower=ratio[0],
        upper=ratio[1],
        interp=interp,
        mean=normalize_mean,
        std=normalize_std,
        output_dtype=output_dtype,
        output_layout=output_layout,
        pad_output=pad_output)

    file_root = args.image_root

    if mode == Mode.TRAIN:
        file_root = os.path.join(file_root, 'train')
    else:
        file_root = os.path.join(file_root, 'val')

    pipe = create_dali_pipeline(
        data_dir=file_root,
        batch_size=batch_size,
        is_training=mode == Mode.TRAIN,
        device_id=device_id,
        ops_meta=ops_meta,
        num_threads=num_threads,
        seed=seed + shard_id,
        shard_id=shard_id,
        num_shards=num_shards)
    pipe.build()
    return DALIGenericIterator([pipe], ['data', 'label'], reader_name='Reader')


def build_dataloader(args, mode):
    """
    Build a dataloader to process datasets. Only DALI dataloader is supported now.
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        mode(utils.Mode): Train or eval mode.

    Returns:
        dataloader(nvidia.dali.plugin.paddle.DALIGenericIterator):
            Iteratable outputs of DALI pipeline,
            including "data" and "label" in type of Paddle's Tensor.
    """
    assert mode in Mode, "Dataset mode should be in supported Modes (train or eval)"
    return dali_dataloader(args, mode, paddle.device.get_device())


def dali_synthetic_dataloader(args, device):
    """
    Define a dali dataloader with synthetic data.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        device(int): Id of GPU to load data.
    Outputs:
        DALIGenericIterator(nvidia.dali.plugin.paddle.DALIGenericIterator)
            Iteratable outputs of DALI pipeline,
            including "data" in type of Paddle's Tensor.
    """
    assert "gpu" in device, "gpu training is required for DALI"

    device_id = int(device.split(':')[1])

    batch_size = args.batch_size
    image_shape = args.image_shape
    output_dtype = types.FLOAT16 if args.dali_output_fp16 else types.FLOAT
    num_threads = args.dali_num_threads

    class ExternalInputIterator(object):
        def __init__(self, batch_size, image_shape):
            from cuda import cudart
            n_bytes = int(batch_size * np.prod(image_shape) * 4)
            err, mem = cudart.cudaMallocHost(n_bytes)
            assert err == cudart.cudaError_t.cudaSuccess
            mem_ptr = ctypes.cast(mem, ctypes.POINTER(ctypes.c_float))
            self.synthetic_data = np.ctypeslib.as_array(mem_ptr, shape=(batch_size, *image_shape))
            self.n = args.benchmark_steps

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            if self.i >= self.n:
                self.__iter__()
                raise StopIteration()
            self.i += 1
            return TensorListCPU(self.synthetic_data, is_pinned=True)

    eli = ExternalInputIterator(batch_size, image_shape)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        images = fn.external_source(source=eli, no_copy=True, dtype=output_dtype)
        images = images.gpu()
        pipe.set_outputs(images)
    pipe.build()
    return DALIGenericIterator([pipe], ['data'])
