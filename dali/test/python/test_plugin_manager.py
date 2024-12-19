# Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali as dali
import nvidia.dali.ops as ops
import nvidia.dali.plugin_manager as plugin_manager
import unittest
import os
import numpy as np
import tempfile

test_bin_dir = os.path.dirname(dali.__file__) + "/test"
batch_size = 4
W = 800
H = 600
C = 3


class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            batch.append(np.array(np.random.rand(H, W, C) * 255, dtype=np.uint8))
            labels.append(np.array(np.random.rand(1) * 10, dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__


eii = ExternalInputIterator(batch_size)
iterator = iter(eii)


class CustomPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CustomPipeline, self).__init__(batch_size, num_threads, device_id)
        self.inputs = ops.ExternalSource()
        self.custom_dummy = ops.CustomDummy(device="gpu")

    def define_graph(self):
        self.images = self.inputs()
        custom_dummy_out = self.custom_dummy(self.images.gpu())
        return (self.images, custom_dummy_out)

    def iter_setup(self):
        (images, labels) = iterator.next()
        self.feed_input(self.images, images)


def load_empty_plugin():
    try:
        plugin_manager.load_library(test_bin_dir + "/libdali_customdummyplugin.so")
    except RuntimeError:
        # in conda "libdali_customdummyplugin" lands inside lib/ dir
        plugin_manager.load_library("libdali_customdummyplugin.so")


class TestLoadedPlugin(unittest.TestCase):
    def test_sysconfig_provides_non_empty_flags(self):
        import nvidia.dali.sysconfig as dali_sysconfig

        assert "" != dali_sysconfig.get_include_flags()
        assert "" != dali_sysconfig.get_compile_flags()
        assert "" != dali_sysconfig.get_link_flags()
        assert "" != dali_sysconfig.get_include_dir()
        assert "" != dali_sysconfig.get_lib_dir()

    def test_load_unexisting_library(self):
        with self.assertRaises(RuntimeError):
            plugin_manager.load_library("not_a_dali_plugin.so")

    def test_load_existing_but_not_a_library(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        for _ in range(10):
            tmp.write(b"0xdeadbeef\n")
        tmp.close()
        with self.assertRaises(RuntimeError):
            plugin_manager.load_library(tmp.name)
        os.remove(tmp.name)

    def test_load_custom_operator_plugin(self):
        with self.assertRaises(AttributeError):
            print(ops.CustomDummy)
        load_empty_plugin()
        print(ops.CustomDummy)

    def test_pipeline_including_custom_plugin(self):
        load_empty_plugin()
        pipe = CustomPipeline(batch_size, 1, 0)
        pipe_out = pipe.run()
        print(pipe_out)
        images, output = pipe_out
        output_cpu = output.as_cpu()
        assert len(images) == batch_size
        assert len(output_cpu) == batch_size

        for i in range(len(images)):
            img = images.at(i)
            out = output_cpu.at(i)
            assert img.shape == out.shape
            np.testing.assert_array_equal(img, out)

    def test_python_operator_and_custom_plugin(self):
        load_empty_plugin()
        ops.readers.TFRecord(path="dummy", index_path="dummy", features={})


if __name__ == "__main__":
    unittest.main()
