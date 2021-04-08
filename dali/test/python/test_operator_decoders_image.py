# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import numpy as np
import glob

from nose import SkipTest
from nose.tools import assert_raises

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path
from test_utils import check_output_pattern

class DecoderPipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_threads, device_id, device, use_fast_idct=False, memory_stats=False):
        super(DecoderPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=1)
        self.input = ops.readers.File(file_root = data_path,
                                      shard_id = 0,
                                      num_shards = 1)
        self.decode = ops.decoders.Image(device = device, output_type = types.RGB, use_fast_idct=use_fast_idct,
                                         memory_stats=memory_stats)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        output = self.decode(inputs)
        return (output, labels)

test_data_root = get_dali_extra_path()
good_path = 'db/single'
missnamed_path = 'db/single/missnamed'
test_good_path = {'jpeg', 'mixed', 'png', 'tiff', 'pnm', 'bmp', 'jpeg2k'}
test_missnamed_path = {'jpeg', 'png', 'tiff', 'pnm', 'bmp'}

def run_decode(data_path, batch, device, threads, memory_stats=False):
    pipe = DecoderPipeline(data_path=data_path, batch_size=batch, num_threads=threads, device_id=0, device=device, memory_stats=memory_stats)
    pipe.build()
    iters = pipe.epoch_size("Reader")
    for _ in range(iters):
        pipe.run()

def test_image_decoder():
    for device in {'cpu', 'mixed'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_good_path:
                    data_path = os.path.join(test_data_root, good_path, img_type)
                    run_decode(data_path, size, device, threads)
                    yield check, img_type, size, device, threads

def test_missnamed_host_decoder():
    for decoder in {'cpu', 'mixed'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_missnamed_path:
                    data_path = os.path.join(test_data_root, missnamed_path, img_type)
                    run_decode(data_path, size, decoder, threads)
                    yield check, img_type, size, decoder, threads

def check(img_type, size, device, threads):
    pass

class DecoderPipelineFastIDC(Pipeline):
    def __init__(self, data_path, batch_size, num_threads, use_fast_idct=False):
        super(DecoderPipelineFastIDC, self).__init__(batch_size, num_threads, 0, prefetch_queue_depth=1)
        self.input = ops.readers.File(file_root = data_path,
                                      shard_id = 0,
                                      num_shards = 1)
        self.decode = ops.decoders.Image(device = 'cpu', output_type = types.RGB, use_fast_idct=use_fast_idct)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        output = self.decode(inputs)
        return (output, labels)

def check_FastDCT_body(batch_size, img_type, device):
    data_path = os.path.join(test_data_root, good_path, img_type)
    compare_pipelines(DecoderPipeline(data_path=data_path, batch_size=batch_size, num_threads=3,
                                      device_id=0, device=device, use_fast_idct=False),
                      DecoderPipeline(data_path=data_path, batch_size=batch_size, num_threads=3,
                                      device_id=0, device='cpu', use_fast_idct=True),
                      # average difference should be no bigger by off-by-3
                      batch_size=batch_size, N_iterations=3, eps=3)

def test_FastDCT():
    for device in {'cpu', 'mixed'}:
        for batch_size in {1, 8}:
            for img_type in test_good_path:
              yield check_FastDCT_body, batch_size, img_type, device

def test_image_decoder_memory_stats():
    device = 'mixed'
    img_type = 'jpeg'
    def check(img_type, size, device, threads):
        data_path = os.path.join(test_data_root, good_path, img_type)
        # largest allocation should match our (in this case) memory padding settings
        # (assuming no reallocation was needed here as the hint is big enough)
        pattern = 'Device memory: \d+ allocations, largest = 16777216 bytes\n' + \
                  'Host \(pinned\) memory: \d+ allocations, largest = 8388608 bytes\n'
        with check_output_pattern(pattern):
            run_decode(data_path, size, device, threads, memory_stats=True)

    for threads in {1, 2, 3, 4}:
        for size in {1, 10}:
            yield check, img_type, size, device, threads

batch_size_test = 16

@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def img_decoder_pipe(device, out_type, files):
    encoded, _ = fn.readers.file(files=files)
    decoded = fn.decoders.image(encoded, device=device, output_type=out_type)
    return decoded

def _testimpl_image_decoder_consistency_multichannel(files):
    compare_pipelines(img_decoder_pipe("cpu", out_type=types.RGB, files=files),
                      img_decoder_pipe("mixed", out_type=types.RGB, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps = 1e-03)
    compare_pipelines(img_decoder_pipe("cpu", out_type=types.ANY_DATA, files=files),
                      img_decoder_pipe("mixed", out_type=types.ANY_DATA, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps = 1e-03)

def test_image_decoder_consistency_multichannel_tiff():
    files = glob.glob(os.path.join(test_data_root, "db/single/multichannel/tiff_multichannel") + "/*.tif*")
    _testimpl_image_decoder_consistency_multichannel(files)

def test_image_decoder_consistenty_multichannel_png_with_alpha():
    files = glob.glob(os.path.join(test_data_root, "db/single/multichannel/with_alpha") + "/*.[!txt]*")
    _testimpl_image_decoder_consistency_multichannel(files)

def _testimpl_image_decoder_consistency(files):
    compare_pipelines(img_decoder_pipe("cpu", out_type=types.ANY_DATA, files=files),
                      img_decoder_pipe("mixed", out_type=types.ANY_DATA, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps = 1)  # differences between JPEG CPU and Mixed backends
    compare_pipelines(img_decoder_pipe("cpu", out_type=types.RGB, files=files),
                      img_decoder_pipe("mixed", out_type=types.RGB, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps = 1)  # differences between JPEG CPU and Mixed backends
    compare_pipelines(img_decoder_pipe( "cpu", out_type=types.GRAY, files=files),
                      img_decoder_pipe("mixed", out_type=types.GRAY, files=files),
                      batch_size=batch_size_test, N_iterations=3,
                      eps = 1)  # differences between JPEG CPU and Mixed backends

def test_image_decoder_consistenty_jpeg():
    files = glob.glob(os.path.join(test_data_root, "db/single/jpeg/113") + "/*.jpg*")
    _testimpl_image_decoder_consistency(files)

def test_image_decoder_consistenty_jpeg2k():
    files = glob.glob(os.path.join(test_data_root, "db/single/jpeg2k/0") + "/*.jp2*")
    _testimpl_image_decoder_consistency(files)

def test_image_decoder_consistenty_bmp():
    files = glob.glob(os.path.join(test_data_root, "db/single/bmp/0") + "/*.bmp*")
    _testimpl_image_decoder_consistency(files)

def test_image_decoder_consistenty_png():
    files = glob.glob(os.path.join(test_data_root, "db/single/png/0") + "/*.png*")
    _testimpl_image_decoder_consistency(files)

@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def decoder_pipe(decoder_op, file_root, device, use_fast_idct):
    encoded, _ = fn.readers.file(file_root=file_root)
    decoded = decoder_op(encoded, device=device, output_type=types.RGB, use_fast_idct=use_fast_idct,
                         seed=42)
    return decoded

def check_image_decoder_alias(new_op, old_op, file_root, device, use_fast_idct):
    new_pipe = decoder_pipe(new_op, file_root, device, use_fast_idct)
    legacy_pipe = decoder_pipe(old_op, file_root, device, use_fast_idct)
    compare_pipelines(new_pipe, legacy_pipe, batch_size=batch_size_test, N_iterations=3)


def test_image_decoder_alias():
    data_path = os.path.join(test_data_root, good_path, "jpeg")
    for new_op, old_op in [(fn.decoders.image, fn.image_decoder),
                           (fn.decoders.image_crop, fn.image_decoder_crop),
                           (fn.decoders.image_random_crop, fn.image_decoder_random_crop)]:
        for device in ["cpu", "mixed"]:
            for use_fast_idct in [True, False]:
                yield check_image_decoder_alias, new_op, old_op, data_path, device, use_fast_idct

@pipeline_def(batch_size=batch_size_test, device_id=0, num_threads=4)
def decoder_slice_pipe(decoder_op, file_root, device, use_fast_idct):
    encoded, labels = fn.readers.file(file_root=file_root)
    start = types.Constant(np.array([0., 0.]))
    end = types.Constant(np.array([0.5, 0.5]))
    decoded = decoder_op(encoded, start, end, device=device,
                         output_type=types.RGB, use_fast_idct=use_fast_idct)
    return decoded


def check_image_decoder_slice_alias(new_op, old_op, file_root, device, use_fast_idct):
    new_pipe = decoder_slice_pipe(new_op, file_root, device, use_fast_idct)
    legacy_pipe = decoder_slice_pipe(old_op, file_root, device, use_fast_idct)
    compare_pipelines(new_pipe, legacy_pipe, batch_size=batch_size_test, N_iterations=3)

def test_image_decoder_slice_alias():
    data_path = os.path.join(test_data_root, good_path, "jpeg")
    new_op, old_op = fn.decoders.image_slice, fn.image_decoder_slice
    for device in ["cpu", "mixed"]:
        for use_fast_idct in [True, False]:
            yield check_image_decoder_slice_alias, new_op, old_op, data_path, device, use_fast_idct
