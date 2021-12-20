# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import numpy as np
from nvidia.dali import fn
from test_utils import get_dali_extra_path
from nose_utils import raises

test_data_root = get_dali_extra_path()


def _uint8_tensor_to_string(t):
    return np.array(t).tobytes().decode()


@pipeline_def
def file_properties(files, device):
    read, _ = fn.readers.file(files=files)
    if device == 'gpu':
        read = read.gpu()
    return fn.get_property(read, key="source_info")


def _test_file_properties(device):
    root_path = os.path.join(test_data_root, 'db', 'single', 'png', '0')
    files = [os.path.join(root_path, i) for i in os.listdir(root_path)]
    p = file_properties(files, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for source_info, ref in zip(out, files):
            assert _uint8_tensor_to_string(source_info) == ref


def test_file_properties():
    for dev in ['cpu', 'gpu']:
        yield _test_file_properties, dev


@pipeline_def
def wds_properties(root_path, device):
    read = fn.readers.webdataset(paths=[root_path], ext=['jpg'])
    if device == 'gpu':
        read = read.gpu()
    return fn.get_property(read, key="source_info")


def _test_wds_properties(device):
    root_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    ref_offset = [1536, 4096, 6144, 8704, 11264, 13824, 16384, 18432]
    p = wds_properties(root_path, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for source_info, offset in zip(out, ref_offset):
            assert _uint8_tensor_to_string(
                source_info) == f"archive {root_path}tar file at \"{root_path}\"component offset {offset}"


def test_wds_properties():
    for dev in ['cpu', 'gpu']:
        yield _test_wds_properties, dev


@pipeline_def
def tfr_properties(root_path, index_path, device):
    import nvidia.dali.tfrecord as tfrec
    inputs = fn.readers.tfrecord(path=root_path, index_path=index_path,
                                 features={"image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                                           "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    enc, lab = fn.get_property(inputs["image/encoded"], key="source_info"), \
               fn.get_property(inputs["image/class/label"], key="source_info")
    if device == 'gpu':
        enc = enc.gpu()
        lab = lab.gpu()
    return enc, lab


def _test_tfr_properties(device):
    root_path = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    index_path = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    idx = [0, 171504, 553687, 651500, 820966, 1142396, 1380096, 1532947]
    p = tfr_properties(root_path, index_path, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out in output:
        out = out if device == 'cpu' else out.as_cpu()
        for source_info, ref_idx in zip(out, idx):
            assert _uint8_tensor_to_string(source_info) == f"{root_path} at index {ref_idx}"


def test_tfr_properties():
    for dev in ['cpu', 'gpu']:
        yield _test_tfr_properties, dev


@pipeline_def
def es_properties(layouts, device):
    num_outputs = len(layouts)

    def gen_data():
        yield np.random.rand(num_outputs, 3, 4, 5)

    inp = fn.external_source(source=gen_data, layout=layouts, num_outputs=num_outputs, batch=False, cycle=True,
                             device=device)
    return tuple(fn.get_property(i, key="layout") for i in inp)


def _test_es_properties(device):
    layouts = ['ABC', 'XYZ']
    p = es_properties(layouts, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()
    for out, lt in zip(output, layouts):
        out = out if device == 'cpu' else out.as_cpu()
        for sample in out:
            assert _uint8_tensor_to_string(sample), lt


def test_es_properties():
    for dev in ['cpu', 'gpu']:
        yield _test_es_properties, dev


@pipeline_def
def improper_property(root_path, device):
    read = fn.readers.webdataset(paths=[root_path], ext=['jpg'])
    return fn.get_property(read, key=["this key doesn't exist"])


@raises(RuntimeError, glob="Unknown property key*")
def _test_improper_property(device):
    root_path = os.path.join(get_dali_extra_path(), "db/webdataset/MNIST/devel-0.tar")
    p = improper_property(root_path, device, batch_size=8, num_threads=4, device_id=0)
    p.build()
    output = p.run()


def test_improper_property():
    for dev in ['cpu', 'gpu']:
        yield _test_improper_property, dev
