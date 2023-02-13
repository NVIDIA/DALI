#  Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from test_utils import get_dali_extra_path, to_array
import os
import numpy as np


test_data_root = get_dali_extra_path()
data_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def compare(lh_out, rh_out):
    assert len(lh_out) == len(rh_out)  # Number of outputs
    for lh_tl, rh_tl in zip(lh_out, rh_out):
        assert len(lh_tl) == len(rh_tl)  # Number of samples
        for lh_tensor, rh_tensor in zip(lh_tl, rh_tl):
            assert np.all(to_array(lh_tensor) == to_array(rh_tensor))  # Each sample


@pipeline_def
def simple_pipe_input(enc, device):
    img = fn.decoders.image(enc, device=device)
    return img


@pipeline_def
def simple_pipe_input_ref(device):
    enc = fn.external_source(name="enc")
    img = fn.decoders.image(enc, device=device)
    return img


def test_simple_pipe_input():
    batch_size = 5
    test_pipe = simple_pipe_input(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1, device='mixed')
    ref_pipe = simple_pipe_input_ref(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1, device='mixed')
    test_pipe.build()
    ref_pipe.build()

    # Load test data
    root_dir = os.path.join(get_dali_extra_path(), 'db', 'single', 'jpeg', '312')
    filenames = [
        "cricket-1345065_1280.jpg", "grasshopper-4357903_1280.jpg", "grasshopper-4357907_1280.jpg"
    ]
    test_data = [np.fromfile(os.path.join(root_dir,filename), dtype=np.uint8) for filename in filenames]

    out = test_pipe.run(enc=test_data)

    ref_pipe.feed_input("enc", test_data)
    ref = ref_pipe.run()
    compare(out, ref)


























def other_func(somearg, device, output_type=types.GRAY):
    img3 = fn.decoders.image(somearg, device=device, output_type=output_type)
    return img3


test_data = [1, 2, 3, 4]


@pipeline_def
def pipeline_def_under_test(enc1, device, enc2):
    img = fn.decoders.image(enc1, device=device)
    img2 = fn.decoders.image(enc2, device=device)
    img2 = fn.resize(img2, size=(200, 200))
    # TODO img3 = other_func(enc3, device) & return img3
    # TODO call function from different module
    return img, img2


@pipeline_def
def reference_pipeline_def(device):
    enc1 = fn.external_source(name='enc1', cycle=False, cuda_stream=1, use_copy_kernel=False,
                              blocking=False, no_copy=True, batch=True, batch_info=False,
                              parallel=False)
    enc2 = fn.external_source(name='enc2', cycle=False, cuda_stream=1, use_copy_kernel=False,
                              blocking=False, no_copy=True, batch=True, batch_info=False,
                              parallel=False)
    img = fn.decoders.image(enc1, device=device)
    img2 = fn.decoders.image(enc2, device=device)
    img2 = fn.resize(img2, size=(200, 200))
    # TODO img3 = other_func(enc3, device) & return img3
    # TODO call function from different module
    return img, img2


def test_not_so_simple_pipeline_input():
    batch_size = 5
    test_pipe = pipeline_def_under_test(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1, device='mixed')
    ref_pipe = reference_pipeline_def(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1, device='mixed')
    test_pipe.build()
    ref_pipe.build()

    # Load test data
    root_dir = os.path.join(get_dali_extra_path(), 'db', 'single', 'jpeg', '312')
    filenames = [
        "cricket-1345065_1280.jpg", "grasshopper-4357903_1280.jpg", "grasshopper-4357907_1280.jpg"
    ]
    test_data = [np.fromfile(os.path.join(root_dir,filename), dtype=np.uint8) for filename in filenames]

    # test_pipe.feed_input("enc", test_data)
    out = test_pipe.run(enc1=test_data, enc2=test_data)

    ref_pipe.feed_input("enc1", test_data)
    ref_pipe.feed_input("enc2", test_data)
    ref = ref_pipe.run()
    compare(out, ref)