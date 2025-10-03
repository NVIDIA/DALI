# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as dali2
import nvidia.dali.backend as _backend
from nose_utils import SkipTest, assert_raises, attr
from test_utils import get_dali_extra_path
import numpy as np
import os

test_data_root = get_dali_extra_path()

def test_tensor_creation_with_device_string_gpu():
    t = dali2.Tensor(np.array([1, 2, 3]), device="gpu")
    assert t.device.device_type == "gpu"
    assert t.device.device_id == 0

def test_tensor_creation_with_device_gpu_object():
    t = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu"))
    assert t.device.device_type == "gpu"
    assert t.device.device_id == 0

def test_tensor_creation_with_device_string_cpu():
    t = dali2.Tensor(np.array([1, 2, 3]), device="cpu")
    assert t.device.device_type == "cpu"

def test_tensor_creation_with_device_cpu_object():
    t = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    assert t.device.device_type == "cpu"

def test_tensor_creation_with_device_gpu_object_variants():
    arr = np.array([1, 2, 3])
    t1 = dali2.Tensor(arr, device=dali2.Device("gpu:0"))
    assert t1.device.device_type == "gpu"
    assert t1.device.device_id == 0
    t2 = dali2.Tensor(arr, device=dali2.Device("gpu"))
    assert t2.device.device_type == "gpu"
    assert t2.device.device_id == 0
    t3 = dali2.Tensor(arr, device=dali2.Device("gpu", 0))
    assert t3.device.device_type == "gpu"
    assert t3.device.device_id == 0


def test_tensor_creation_with_device_cuda_object_variants():
    arr = np.array([1, 2, 3])
    t1 = dali2.Tensor(arr, device=dali2.Device("cuda:0"))
    assert t1.device.device_type == "gpu"
    assert t1.device.device_id == 0
    t2 = dali2.Tensor(arr, device=dali2.Device("cuda"))
    assert t2.device.device_type == "gpu"
    assert t2.device.device_id == 0
    t3 = dali2.Tensor(arr, device=dali2.Device("cuda", 0))
    assert t3.device.device_type == "gpu"
    assert t3.device.device_id == 0


def test_tensor_creation_with_device_cpu_object_variants():
    arr = np.array([1, 2, 3])
    t1 = dali2.Tensor(arr, device=dali2.Device("cpu"))
    assert t1.device.device_type == "cpu"


def test_tensor_addition_with_unanimous_gpu_inputs():
    gpu_tensor1 = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu"))
    gpu_tensor2 = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device("gpu"))
    result = gpu_tensor1 + gpu_tensor2
    assert result.device.device_type == "gpu"


def test_tensor_addition_with_unanimous_cpu_inputs():
    cpu_tensor1 = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    cpu_tensor2 = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device("cpu"))
    result = cpu_tensor1 + cpu_tensor2
    assert result.device.device_type == "cpu"


def test_tensor_addition_with_mixed_cpu_gpu_inputs_raises_error():
    cpu_tensor = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    gpu_tensor = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device("gpu"))
    # This should raise an error when device=None and inputs have different devices
    with assert_raises(RuntimeError, glob="*not on the requested device*"):
        cpu_tensor + gpu_tensor


def test_tensor_addition_with_unanimous_gpu_ordinals():
    for device_id in range(_backend.GetCUDADeviceCount()):
        gpu_tensor1 = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device(f"gpu:{device_id}"))
        gpu_tensor2 = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device(f"gpu:{device_id}"))
        result = gpu_tensor1 + gpu_tensor2
        assert result.device.device_type == "gpu"
        assert result.device.device_id == device_id


@attr("multi_gpu")
def test_tensor_addition_with_mixed_gpu_ordinals_raises_error():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    gpu_tensor1 = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu:0"))
    gpu_tensor2 = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device("gpu:1"))
    with assert_raises(RuntimeError, glob="*incompatible device*"):
        gpu_tensor1 + gpu_tensor2


def test_slice_operator_with_gpu_inputs_infers_gpu_device():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu:0"))
    anchor = dali2.Tensor(
        [
            1,
        ],
        device=dali2.Device("gpu:0"),
    )
    shape = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("gpu:0"),
    )
    result = dali2.slice(data, anchor, shape, axes=(0,))
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


def test_slice_operator_with_cpu_inputs_infers_cpu_device():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    anchor = dali2.Tensor(
        [
            1,
        ],
        device=dali2.Device("cpu"),
    )
    shape = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("cpu"),
    )
    result = dali2.slice(data, anchor, shape, axes=(0,))
    assert result.device.device_type == "cpu"


def test_slice_operator_with_gpu_inputs_and_device_none_infers_gpu_device():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu:0"))
    anchor = dali2.Tensor(
        [
            1,
        ],
        device=dali2.Device("gpu:0"),
    )
    shape = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("gpu:0"),
    )
    result = dali2.slice(data, anchor, shape, axes=(0,), device=None)
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


def test_slice_operator_with_cpu_inputs_and_device_none_infers_cpu_device():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    anchor = dali2.Tensor(
        [
            1,
        ],
        device=dali2.Device("cpu"),
    )
    shape = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("cpu"),
    )
    result = dali2.slice(data, anchor, shape, axes=(0,), device=None)
    assert result.device.device_type == "cpu"


def test_slice_operator_with_cpu_inputs_and_gpu_device_raises_error():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cpu"))
    anchor = dali2.Tensor(
        [
            1,
        ],
        device=dali2.Device("cpu"),
    )
    shape = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("cpu"),
    )
    with assert_raises(RuntimeError, glob="*incompatible device*"):
        result = dali2.slice(data, anchor, shape, device=dali2.Device("gpu:0"), axes=(0,))


def test_slice_operator_kwargs_must_be_cpu_tensors():
    data = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("gpu:0"))
    end = dali2.Tensor(
        [
            2,
        ],
        device=dali2.Device("gpu:0"),
    )
    with assert_raises(TypeError, glob="*'end'*"):
        result = dali2.slice(data, device=dali2.Device("gpu:0"), end=end)
        res_cpu = result.cpu()


def test_image_decoder_defaults_to_cpu_device():
    image_path = os.path.join(test_data_root, "db", "single", "jpeg", "100", "swan-3584559_640.jpg")
    raw_bytes = np.fromfile(image_path, dtype=np.uint8)
    result = dali2.decoders.image(raw_bytes)
    assert result.device.device_type == "cpu"


def test_image_decoder_with_explicit_mixed_device():
    print("TODO(janton): to be removed")
    image_path = os.path.join(test_data_root, "db", "single", "jpeg", "100", "swan-3584559_640.jpg")
    raw_bytes = np.fromfile(image_path, dtype=np.uint8)
    result = dali2.decoders.image(raw_bytes, device="mixed")
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


def test_image_decoder_with_explicit_gpu_device():
    raise SkipTest("TODO(janton): Not yet supported")
    image_path = os.path.join(test_data_root, "db", "single", "jpeg", "100", "swan-3584559_640.jpg")
    raw_bytes = np.fromfile(image_path, dtype=np.uint8)
    result = dali2.decoders.image(raw_bytes, device=dali2.Device("gpu:0"))
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


def test_file_reader_defaults_to_cpu_device():
    reader = dali2.readers.File(file_root=os.path.join(test_data_root, "db", "single", "jpeg"))
    count = 0
    for data, label in reader.samples():
        assert data.device.device_type == "cpu"
        assert label.device.device_type == "cpu"
        count += 1
        if count >= 5:
            break


def test_file_reader_with_gpu_device_raises_error():
    with assert_raises(RuntimeError, glob="*not registered for gpu*"):
        reader = dali2.readers.File(
            file_root=os.path.join(test_data_root, "db", "single", "jpeg"),
            device=dali2.Device("gpu:0"),
        )
        for data, label in reader.samples():
            pass


def test_tensor_addition_with_cuda_string_variations():
    # Test 'cuda:0' vs 'gpu:0' equivalence
    gpu_tensor1 = dali2.Tensor(np.array([1, 2, 3]), device=dali2.Device("cuda:0"))
    gpu_tensor2 = dali2.Tensor(np.array([4, 5, 6]), device=dali2.Device("gpu:0"))
    result1 = gpu_tensor1 + gpu_tensor2
    assert result1.device.device_type == "gpu"
    assert result1.device.device_id == 0


def test_uniform_operator_device_defaults_to_cpu():
    # Test that the uniform operator produces tensors on the correct device
    uniform_op = dali2.ops.random.Uniform(max_batch_size=8)
    result = uniform_op()
    assert result.device.device_type == "cpu"


def test_uniform_operator_device_explicit_cpu():
    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="cpu")
    result = uniform_op()
    assert result.device.device_type == "cpu"


def test_uniform_operator_device_explicit_gpu():
    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="gpu")
    result = uniform_op()
    assert result.device.device_type == "gpu"


def test_uniform_operator_device_explicit_gpu_ordinal():
    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="gpu:0")
    result = uniform_op()
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0

    if _backend.GetCUDADeviceCount() >= 2:
        uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="gpu:1")
        result = uniform_op()
        assert result.device.device_type == "gpu"
        assert result.device.device_id == 1


def test_uniform_operator_device_explicit_cuda_ordinal():
    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="cuda:0")
    result = uniform_op()
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0

    if _backend.GetCUDADeviceCount() >= 2:
        uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="cuda:1")
        result = uniform_op()
        assert result.device.device_type == "gpu"
        assert result.device.device_id == 1


def test_video_reader_device_inference():
    video_path = os.path.join(
        test_data_root, "db", "video", "sintel", "video_files", "sintel_trailer-720p_0.mp4"
    )
    with assert_raises(RuntimeError, glob="*not registered for cpu*"):
        video_reader = dali2.readers.Video(filenames=[video_path], sequence_length=10)
        (result,) = next(video_reader.samples())


def test_video_reader_device_explicit_gpu():
    video_path = os.path.join(
        test_data_root, "db", "video", "sintel", "video_files", "sintel_trailer-720p_0.mp4"
    )
    video_reader = dali2.readers.Video(filenames=[video_path], sequence_length=10, device="gpu")
    (result,) = next(video_reader.samples())
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


def test_video_decoder_device_inference():
    video_path = os.path.join(
        test_data_root, "db", "video", "sintel", "video_files", "sintel_trailer-720p_0.mp4"
    )
    encoded_video = np.fromfile(video_path, dtype=np.uint8)
    decoded = dali2.experimental.decoders.video(encoded_video)
    assert decoded.device.device_type == "cpu"


def test_video_decoder_device_explicit_mixed():
    video_path = os.path.join(
        test_data_root, "db", "video", "sintel", "video_files", "sintel_trailer-720p_0.mp4"
    )
    encoded_video = np.fromfile(video_path, dtype=np.uint8)
    decoded = dali2.experimental.decoders.video(
        encoded_video, start_frame=0, sequence_length=10, device="mixed"
    )
    assert decoded.device.device_type == "gpu"
    assert decoded.device.device_id == 0


def test_video_decoder_device_explicit_gpu():
    raise SkipTest("TODO(janton): Not yet supported")
    video_path = os.path.join(
        test_data_root, "db", "video", "sintel", "video_files", "sintel_trailer-720p_0.mp4"
    )
    encoded_video = np.fromfile(video_path, dtype=np.uint8)
    decoded = dali2.experimental.decoders.video(encoded_video, sequence_length=10, device="gpu")
    assert decoded.device.device_type == "gpu"
    assert decoded.device.device_id == 0


@attr("pytorch")
def test_operator_with_torch_device_cpu():
    import torch

    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device=torch.device("cpu"))
    result = uniform_op()
    assert result.device.device_type == "cpu"


@attr("pytorch")
def test_operator_with_torch_device_gpu():
    import torch

    uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device=torch.device("cuda"))
    result = uniform_op()
    assert result.device.device_type == "gpu"
    assert result.device.device_id == 0


@attr("pytorch")
def test_operator_with_torch_device_gpu_ordinal():
    import torch

    for device_id in range(_backend.GetCUDADeviceCount()):
        uniform_op = dali2.ops.random.Uniform(
            max_batch_size=8, device=torch.device(f"cuda:{device_id}")
        )
        result = uniform_op()
        assert result.device.device_type == "gpu"
        assert result.device.device_id == device_id


def test_operator_invalid_device_string():
    with assert_raises(ValueError, glob="*Invalid device*"):
        uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="invalid")


def test_operator_invalid_device_string_ordinal():
    with assert_raises(ValueError, glob="*Invalid device*"):
        uniform_op = dali2.ops.random.Uniform(max_batch_size=8, device="gpu:999")
