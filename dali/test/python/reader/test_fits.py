# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import os
from astropy.io import fits
import numpy as np
import tempfile
import random
from test_utils import to_array
from numpy.testing import assert_array_equal

rng = np.random.RandomState(12345)


def create_fits_file(filename, shape, type=np.int32, compressed=False, hdus=1):
    hdu_list = [fits.PrimaryHDU(header=None)]
    for i in range(hdus):
        data = rng.randint(100, size=shape).astype(type)
        hdu = fits.ImageHDU(data, name=f"IMAGE{i + 1}")
        if compressed:
            hdu = fits.CompImageHDU(data, name=f"IMAGE{i + 1}")
        hdu_list.append(hdu)

    hdulist = fits.HDUList(hdu_list)
    hdulist.writeto(filename, overwrite=True)


@pipeline_def
def FitsReaderPipeline(path, device="cpu", file_list=None, files=None, file_filter="*.fits",
                       hdu_indices=[2], dtype=None):
    data = fn.experimental.readers.fits(device=device, file_list=file_list, files=files,
                                        file_root=path, file_filter=file_filter, shard_id=0,
                                        num_shards=1, hdu_indices=hdu_indices)

    return tuple(data) if type(data) is list else data


supported_numpy_types = set([
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
    np.uint32, np.uint64, np.float32, np.float64,
])

# Astropy doesn't support writing 64 bit types to compressed images
# Furthermore, currently we don't handle accelerated float decompression
unsupported_compression_numpy_types = set([
    np.float32,
    np.float64,
    np.int64,
    np.uint64,
])

# Test shapes, for each number of dims, astropy & fits do not handle dims = ()
test_shapes = {
    1: [(10,), (12,), (10,), (20,), (10,), (12,), (13,), (19,)],
    2: [(10, 10), (12, 10), (10, 12), (20, 15), (10, 11), (12, 11), (13, 11), (19, 10)],
    3: [(6, 2, 5), (5, 6, 2), (3, 3, 3), (10, 1, 8), (8, 8, 3), (2, 2, 3), (8, 4, 3), (1, 10, 1)],
    4: [(2, 6, 2, 5), (5, 1, 6, 2), (3, 2, 3, 3), (1, 10, 1, 8), (2, 8, 2, 3), (2, 3, 2, 3),
        (1, 8, 4, 3), (1, 3, 10, 1)],
}


def _testimpl_types_and_shapes(device, shapes, num_outputs, type, batch_size, num_threads,
                               compressed_arg, file_arg_type):
    """ compare reader with astropy, with different batch_size and num_threads """

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
            create_fits_file(full_paths[i], shapes[i], type, compressed, num_outputs)

        # load manually, we skip primary HDU since it only stores metadata
        # astropy returns data from each HDUs as a ndarray
        hduls = [fits.open(filename) for filename in full_paths]
        arrays = [hdu.data for hdul in hduls for hdu in hdul[1:]]
        hdu_indices = list(range(2, num_outputs + 2))

        # load with numpy reader
        file_list_arg = None
        files_arg = None
        file_filter_arg = None
        if file_arg_type == 'file_list':
            file_list_arg = os.path.join(test_data_root, "input.lst")
            with open(file_list_arg, "w") as f:
                f.writelines("\n".join(filenames))
        elif file_arg_type == 'files':
            files_arg = filenames
        elif file_arg_type == "file_filter":
            file_filter_arg = "*.fits"
        else:
            assert False

        pipe = FitsReaderPipeline(path=test_data_root, files=files_arg, file_list=file_list_arg,
                                  file_filter=file_filter_arg, device=device, batch_size=batch_size,
                                  num_threads=num_threads, device_id=0, hdu_indices=hdu_indices)

        try:
            pipe.build()
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


def test_reading_uncompressed():
    compressed = False
    for device in ["cpu", "gpu"]:
        for type in supported_numpy_types:
            for ndim in test_shapes.keys():
                shapes = test_shapes[ndim]
                file_arg_type = random.choice(['file_list', 'files', 'file_filter'])
                num_threads = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
                batch_size = random.choice([1, 3, 4, 8, 16, 32, 64, 128])
                num_outputs = random.choice([1, 3, 4, 8])
                yield _testimpl_types_and_shapes, device, shapes, num_outputs, type, batch_size, \
                    num_threads, compressed, file_arg_type,


def test_reading_compressed():
    compressed = True
    for device in ["cpu", "gpu"]:
        for type in supported_numpy_types - unsupported_compression_numpy_types:
            for ndim in test_shapes.keys():
                if ndim > 3:  # astropy doesn't support compression of images with more dimensions
                    continue
                shapes = test_shapes[ndim]
                file_arg_type = random.choice(['file_list', 'files', 'file_filter'])
                num_threads = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
                batch_size = random.choice([1, 3, 4, 8, 16, 32, 64, 128])
                num_outputs = random.choice([1, 3, 4, 8])
                yield _testimpl_types_and_shapes, device, shapes, num_outputs, type, batch_size, \
                    num_threads, compressed, file_arg_type,
