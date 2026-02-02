# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tempfile
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
import numpy as np
from ndd_vs_fn_test_utils import N_ITERATIONS, run_operator_test
from test_ndd_vs_fn_readers import run_reader_test


def setup_test_numpy_reader_cpu():
    tmp_dir = tempfile.TemporaryDirectory()
    dir_name = tmp_dir.name

    rng = np.random.default_rng(12345)

    def create_numpy_file(filename, shape, typ, fortran_order):
        # generate random array
        arr = rng.random(shape) * 10.0
        arr = arr.astype(typ)
        if fortran_order:
            arr = np.asfortranarray(arr)
        np.save(filename, arr)

    num_samples = 20
    filenames = []
    for index in range(0, num_samples):
        filename = os.path.join(dir_name, "test_{:02d}.npy".format(index))
        filenames.append(filename)
        create_numpy_file(filename, (5, 2, 8), np.float32, False)

    return tmp_dir


@params("cpu")
def test_numpy_reader(device):
    with setup_test_numpy_reader_cpu() as test_data_root:
        run_reader_test(
            fn_reader=fn.readers.numpy,
            ndd_reader=ndd.readers.Numpy,
            device=device,
            reader_args={"file_root": test_data_root},
        )


@params("cpu")
def test_numpy_decoder(device):
    with setup_test_numpy_reader_cpu() as test_data_root:
        file_list = os.listdir(test_data_root)
        data = [
            [np.fromfile(os.path.join(test_data_root, f), dtype=np.uint8) for f in file_list]
        ] * N_ITERATIONS
        run_operator_test(
            input_epoch=data,
            fn_operator=fn.decoders.numpy,
            ndd_operator=ndd.decoders.numpy,
            device=device,
        )


tested_operators = [
    "readers.numpy",
    "decoders.numpy",
]
