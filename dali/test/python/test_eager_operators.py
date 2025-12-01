# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import os

from nvidia.dali import fn
from nvidia.dali import pipeline_def
from nvidia.dali import ops
from nvidia.dali import tensors
from nvidia.dali.experimental import eager
from nose_utils import assert_raises, raises
from test_utils import get_dali_extra_path


@raises(RuntimeError, glob="Argument '*' is not supported by eager operator 'crop'.")
def _test_disqualified_argument(key):
    tl = tensors.TensorListCPU(np.zeros((8, 256, 256, 3)))
    eager.crop(tl, crop=[64, 64], **{key: 0})


def test_disqualified_arguments():
    for arg in ["bytes_per_sample_hint", "preserve", "seed"]:
        yield _test_disqualified_argument, arg


@raises(TypeError, glob="unsupported operand type*")
def test_arithm_op_context_manager_disabled():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))

    tl_1 + tl_2


def test_arithm_op_context_manager_enabled():
    eager.arithmetic(True)
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))

    assert np.array_equal((tl_1 + tl_2).as_array(), np.full(shape=(8, 16, 16), fill_value=2))
    eager.arithmetic(False)


def test_arithm_op_context_manager_nested():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    expected_sum = np.full(shape=(8, 16, 16), fill_value=2)

    with eager.arithmetic():
        assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

        with eager.arithmetic(False):
            with assert_raises(TypeError, glob="unsupported operand type*"):
                tl_1 + tl_2

        assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)


def test_arithm_op_context_manager_deep_nested():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    expected_sum = np.full(shape=(8, 16, 16), fill_value=2)

    eager.arithmetic(True)

    assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

    with eager.arithmetic(False):
        with assert_raises(TypeError, glob="unsupported operand type*"):
            tl_1 + tl_2

        with eager.arithmetic(True):
            np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

            with eager.arithmetic(False):
                with assert_raises(TypeError, glob="unsupported operand type*"):
                    tl_1 + tl_2

        with assert_raises(TypeError, glob="unsupported operand type*"):
            tl_1 + tl_2

    assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)
    eager.arithmetic(False)


def test_identical_rng_states():
    eager_state_1 = eager.rng_state(seed=42)
    eager_state_2 = eager.rng_state(seed=42)

    out_1_1 = eager_state_1.random.normal(shape=[5, 5], batch_size=8)
    out_1_2 = eager_state_1.noise.gaussian(out_1_1)
    out_1_3 = eager_state_1.random.normal(shape=[5, 5], batch_size=8)

    out_2_1 = eager_state_2.random.normal(shape=[5, 5], batch_size=8)
    out_2_2 = eager_state_2.noise.gaussian(out_2_1)
    out_2_3 = eager_state_2.random.normal(shape=[5, 5], batch_size=8)

    assert np.allclose(out_1_1.as_tensor(), out_2_1.as_tensor())
    assert np.allclose(out_1_2.as_tensor(), out_2_2.as_tensor())
    assert np.allclose(out_1_3.as_tensor(), out_2_3.as_tensor())


def test_identical_rng_states_interleaved():
    eager_state_1 = eager.rng_state(seed=42)
    eager_state_2 = eager.rng_state(seed=42)

    out_1_1 = eager_state_1.random.normal(shape=[5, 5], batch_size=8)
    eager_state_1.random.normal(shape=[6, 6], batch_size=8)
    eager_state_1.noise.gaussian(out_1_1)
    out_1_2 = eager_state_1.random.normal(shape=[5, 5], batch_size=8)

    out_2_1 = eager_state_2.random.normal(shape=[5, 5], batch_size=8)
    out_2_2 = eager_state_2.random.normal(shape=[5, 5], batch_size=8)

    assert np.allclose(out_1_1.as_tensor(), out_2_1.as_tensor())
    assert np.allclose(out_1_2.as_tensor(), out_2_2.as_tensor())


def test_objective_eager_resize():
    from nvidia.dali._utils import eager_utils

    resize_class = eager_utils._eager_op_object_factory(
        ops.python_op_factory("Resize", "Resize"), "Resize"
    )
    tl = tensors.TensorListCPU(
        np.random.default_rng().integers(256, size=(8, 200, 200, 3), dtype=np.uint8)
    )

    obj_resize = resize_class(resize_x=50, resize_y=50)
    out_obj = obj_resize(tl)
    out_fun = eager.resize(tl, resize_x=50, resize_y=50)

    assert np.array_equal(out_obj.as_tensor(), out_fun.as_tensor())


@pipeline_def(num_threads=3, device_id=0)
def mixed_image_decoder_pipeline(file_root, seed):
    jpeg, _ = fn.readers.file(file_root=file_root, seed=seed)
    out = fn.decoders.image(jpeg, device="mixed")

    return out


def test_mixed_devices_decoder():
    """Tests hidden functionality of exposing eager operators as classes."""
    seed = 42
    batch_size = 8
    file_root = os.path.join(get_dali_extra_path(), "db/single/jpeg")

    pipe = mixed_image_decoder_pipeline(file_root, seed, batch_size=batch_size)
    (pipe_out,) = pipe.run()

    jpeg, _ = next(eager.readers.file(file_root=file_root, batch_size=batch_size, seed=seed))
    eager_out = eager.decoders.image(jpeg, device="mixed")

    assert len(pipe_out) == len(eager_out)

    with eager.arithmetic():
        for comp_tensor in pipe_out == eager_out:
            assert np.all(comp_tensor.as_cpu())
