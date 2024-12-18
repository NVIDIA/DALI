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

import itertools
import os
import sys

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from astropy.io import fits
import numpy as np
import tempfile
import random
from nose2.tools import params
from test_utils import to_array, get_dali_extra_path
from numpy.testing import assert_array_equal


def enum_product(*params):
    return tuple((i,) + t for i, t in enumerate(itertools.product(*params)))


def create_fits_file(np_rng, filename, shape, type=np.int32, compressed=False, hdus=1):
    hdu_list = [fits.PrimaryHDU(header=None)]
    for i in range(hdus):
        data = np_rng.randint(100, size=shape).astype(type)
        hdu = fits.ImageHDU(data, name=f"IMAGE{i + 1}")
        if compressed:
            hdu = fits.CompImageHDU(data, name=f"IMAGE{i + 1}")
        hdu_list.append(hdu)

    hdulist = fits.HDUList(hdu_list)
    hdulist.writeto(filename, overwrite=True)


@pipeline_def
def FitsReaderPipeline(
    path,
    device="cpu",
    file_list=None,
    files=None,
    file_filter="*.fits",
    hdu_indices=[2],
    dtype=None,
):
    data = fn.experimental.readers.fits(
        device=device,
        file_list=file_list,
        files=files,
        file_root=path,
        file_filter=file_filter,
        shard_id=0,
        num_shards=1,
        hdu_indices=hdu_indices,
    )

    return tuple(data) if type(data) is list else data


def get_dtypes(compression):
    # keep it as a list to retain the declaration order of elements
    all = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ]
    excluded = set()
    # The astropy is not actively developed for Python3.6 and the last available
    # version does not support some dtypes
    vi = sys.version_info
    if vi.major < 3 or (vi.major == 3 and vi.minor <= 7):
        excluded |= {np.int8}
    # Astropy doesn't support writing those types to compressed image
    # Futhermore, due to issues with gpu acceleration of rice decoding of floats,
    # it is not supported yet
    if compression:
        excluded |= {np.int64, np.uint64, np.float32, np.float64}
    return [dtype for dtype in all if dtype not in excluded]


# Test shapes, for each number of dims, astropy & fits do not handle dims = ()
test_shapes = {
    1: [(10,), (12,), (10,), (20,), (10,), (12,), (13,), (19,)],
    2: [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)],
    3: [(6, 2, 5), (5, 6, 2), (3, 3, 3), (10, 1, 8), (8, 8, 3), (2, 2, 3), (8, 4, 3), (1, 10, 1)],
    4: [
        (2, 6, 2, 5),
        (5, 1, 6, 2),
        (3, 2, 3, 3),
        (1, 10, 1, 8),
        (2, 8, 2, 3),
        (2, 3, 2, 3),
        (1, 8, 4, 3),
        (1, 3, 10, 1),
    ],
}


def _testimpl_types_and_shapes(
    np_rng,
    device,
    shapes,
    num_outputs,
    type,
    batch_size,
    num_threads,
    compressed_arg,
    file_arg_type,
):
    """compare reader with astropy, with different batch_size and num_threads"""

    nsamples = len(shapes)
    # setup files
    with tempfile.TemporaryDirectory() as test_data_root:
        # setup file
        filenames = ["test_{:02d}.fits".format(i) for i in range(nsamples)]
        full_paths = [os.path.join(test_data_root, fname) for fname in filenames]
        for i in range(nsamples):
            compressed = compressed_arg
            if compressed is None:
                compressed = random.choice([False, True])
            create_fits_file(np_rng, full_paths[i], shapes[i], type, compressed, num_outputs)

        # load manually, we skip primary HDU since it only stores metadata
        # astropy returns data from each HDUs as a ndarray
        hduls = [fits.open(filename) for filename in full_paths]
        arrays = [hdu.data for hdul in hduls for hdu in hdul[1:]]
        hdu_indices = list(range(2, num_outputs + 2))

        # load with numpy reader
        file_list_arg = None
        files_arg = None
        file_filter_arg = None
        if file_arg_type == "file_list":
            file_list_arg = os.path.join(test_data_root, "input.lst")
            with open(file_list_arg, "w") as f:
                f.writelines("\n".join(filenames))
        elif file_arg_type == "files":
            files_arg = filenames
        elif file_arg_type == "file_filter":
            file_filter_arg = "*.fits"
        else:
            assert False

        pipe = FitsReaderPipeline(
            path=test_data_root,
            files=files_arg,
            file_list=file_list_arg,
            file_filter=file_filter_arg,
            device=device,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=0,
            hdu_indices=hdu_indices,
        )

        try:
            i = 0
            while i < nsamples:
                pipe_out = pipe.run()
                for s in range(batch_size):
                    if i >= nsamples:
                        break
                    for out_idx in range(num_outputs):
                        pipe_arr = to_array(pipe_out[out_idx][s])
                        ref_arr = arrays[i + out_idx]
                        assert_array_equal(pipe_arr, ref_arr)
                    i += num_outputs
        finally:
            del pipe


@params(*enum_product(get_dtypes(False), [1, 2, 3, 4], ["cpu", "gpu"]))
def test_reading_uncompressed(i, dtype, ndim, device):
    rng = np.random.default_rng(42 + i)
    np_rng = np.random.RandomState(12345 + i)
    compressed = False
    shapes = test_shapes[ndim]
    file_arg_type = rng.choice(["file_list", "files", "file_filter"])
    num_threads = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
    batch_size = rng.choice([1, 3, 4, 8, 16])
    num_outputs = rng.choice([1, 3, 4, 8])
    _testimpl_types_and_shapes(
        np_rng,
        device,
        shapes,
        num_outputs,
        dtype,
        batch_size,
        num_threads,
        compressed,
        file_arg_type,
    )


@params(*enum_product(get_dtypes(True), [1, 2, 3], ["cpu", "gpu"]))
def test_reading_compressed(i, dtype, ndim, device):
    assert ndim <= 3  # astropy doesn't support compression of images with more dimensions
    rng = np.random.default_rng(42 + i)
    np_rng = np.random.RandomState(12345 + i)
    compressed = True
    shapes = test_shapes[ndim]
    file_arg_type = rng.choice(["file_list", "files", "file_filter"])
    num_threads = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
    batch_size = rng.choice([1, 3, 4, 8, 16])
    num_outputs = rng.choice([1, 3, 4, 8])
    _testimpl_types_and_shapes(
        np_rng,
        device,
        shapes,
        num_outputs,
        dtype,
        batch_size,
        num_threads,
        compressed,
        file_arg_type,
    )


@params(
    "cpu",
)
def test_concurrent_pipelines(device):
    test_data_path = os.path.join(get_dali_extra_path(), "db/single/fits/compressed/")

    pipelines = []
    for _ in range(5):
        pipeline = FitsReaderPipeline(
            test_data_path, device=device, device_id=0, num_threads=4, batch_size=1
        )
        pipelines.append(pipeline)

    for _ in range(3):
        expected_output = None
        for p in pipelines:
            pipeline_output = p.run()
            if expected_output is None:
                expected_output = pipeline_output
            else:
                assert np.all(pipeline_output[0].as_array() == expected_output[0].as_array())
