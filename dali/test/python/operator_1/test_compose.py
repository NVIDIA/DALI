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

import numpy as np
import os
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import test_utils


def test_unified_arg_placement():
    batch_size = 30

    pipe = Pipeline(batch_size, 1, None)
    with pipe:
        u = ops.random.Uniform()(range=(1, 2), shape=3)
        tr = ops.transforms.Translation(offset=u, device="cpu")
        pipe.set_outputs(tr(), u)
    matrices, offsets = pipe.run()
    assert len(matrices) == batch_size
    for i in range(len(matrices)):
        offset = offsets.at(i)
        matrix = matrices.at(i)
        assert offset.shape == (3,)
        for j in range(3):
            assert offset[j] >= 1 and offset[j] < 2  # check that it's not all zeros or sth
        T = offset[:, np.newaxis]  # convert to a columnn
        assert np.array_equal(matrix, np.concatenate([np.identity(3), T], axis=1))


def test_compose():
    batch_size = 3
    pipe = Pipeline(batch_size, 1, None)

    u = ops.random.Uniform()(range=(1, 2), shape=3)
    c1 = ops.Compose([ops.transforms.Translation(offset=u), ops.transforms.Scale(scale=[1, 1, -1])])
    c2 = ops.Compose([c1, ops.transforms.Rotation(angle=90, axis=[0, 0, 1])])
    pipe.set_outputs(c2(fn.transforms.scale(scale=[2, 2, 2])), u)
    matrices, offsets = pipe.run()
    assert len(matrices) == batch_size
    for i in range(len(matrices)):
        offset = offsets.at(i)
        matrix = matrices.at(i)
        assert offset.shape == (3,)
        for j in range(3):
            assert offset[j] >= 1 and offset[j] < 2  # check that it's not all zeros or sth
        mtx = np.float32([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
        T = offset[:, np.newaxis]  # convert to a columnn
        T = np.dot(mtx, T)
        mtx *= 2
        assert np.allclose(matrix, np.concatenate([mtx, T], axis=1), rtol=1e-5, atol=1e-6)


test_data_root = os.environ["DALI_EXTRA_PATH"]
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")


def test_compose_change_device():
    batch_size = 3
    pipe = Pipeline(batch_size, 1, 0)

    size = fn.random.uniform(shape=2, range=(300, 500))
    c = ops.Compose([ops.decoders.Image(device="cpu"), ops.Resize(size=size, device="gpu")])
    files, labels = fn.readers.caffe(path=caffe_db_folder, seed=1)
    pipe.set_outputs(c(files), fn.resize(fn.decoders.image(files).gpu(), size=size))

    out = pipe.run()
    assert isinstance(out[0], dali.backend.TensorListGPU)
    test_utils.check_batch(out[0], out[1], batch_size=batch_size)
