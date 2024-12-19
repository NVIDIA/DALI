# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def
from nvidia.dali.backend import TensorGPU
import nvidia.dali.fn as fn
import nvidia.dali.types as types


def get_dali_tensor_gpu(value, shape, dtype, device_id=0) -> TensorGPU:
    """Helper function to create DALI TensorGPU.

    Args:
        value : Value to fill the tensor with.
        shape : Shape for the tensor.
        dtype : Data type for the tensor.

    Returns:
        TensorGPU: DALI TensorGPU with provided shape and dtype filled
        with provided value.
    """

    @pipeline_def(num_threads=1, batch_size=1, exec_dynamic=True)
    def dali_pipeline():
        values = types.Constant(value=np.full(shape, value, dtype), device="gpu")

        return values

    pipe = dali_pipeline(device_id=device_id)
    dali_output = pipe.run()

    return dali_output[0][0]


def sequential_pipeline(batch_size, shape):
    """Helper to create DALI pipelines that return GPU tensors with sequential values.

    Args:
        batch_size: Batch size for the pipeline.
        shape : Shape for the output tensor.
    """

    def numpy_sequential_tensors(sample_info):
        return np.full(shape, sample_info.idx_in_epoch, dtype=np.int32)

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def sequential_pipeline_def():
        data = fn.external_source(
            source=numpy_sequential_tensors, num_outputs=1, batch=False, dtype=types.INT32
        )
        data = data[0].gpu()
        return data

    return sequential_pipeline_def()


def pipeline_with_variable_shape_output(batch_size):
    """Helper to create DALI pipelines that return GPU tensors with variable shape.

    Args:
        batch_size: Batch size for the pipeline.
    """

    def numpy_tensors(sample_info):
        tensors = [
            np.full((1, 5), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1, 3), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1, 2), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1, 4), sample_info.idx_in_epoch, dtype=np.int32),
        ]
        return tensors[sample_info.idx_in_epoch % len(tensors)]

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def sequential_pipeline_def():
        data = fn.external_source(
            source=numpy_tensors, num_outputs=1, batch=False, dtype=types.INT32
        )
        data = data[0].gpu()
        return data

    return sequential_pipeline_def()


data_path = os.path.join(os.environ["DALI_EXTRA_PATH"], "db", "single", "jpeg")


def get_all_files_from_directory(dir_path, ext):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list


file_names = get_all_files_from_directory(data_path, ".jpg")
file_labels = [*range(len(file_names))]


def iterator_function_def(shard_id=0, num_shards=1):
    files, labels = fn.readers.file(
        name="reader",
        files=file_names,
        labels=file_labels,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    return labels.gpu()
