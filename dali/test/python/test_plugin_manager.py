# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin_manager as plugin_manager
import unittest

image_dir = "../docs/examples/images"
batch_size = 128

class CustomPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(CustomPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.FileReader(file_root = image_dir)
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
            self.custom_dummy = ops.CustomDummy( device = "gpu")

        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            custom_dummy_out = self.custom_dummy(images.gpu())
            return (images, custom_dummy_out)

        def iter_setup(self):
            pass

class TestLoadedPlugin(unittest.TestCase):
    def test_load_unexisting_library(self):
        with self.assertRaises(RuntimeError):
            plugin_manager.load_library("unexisting.so")

    def test_load_custom_operator_plugin(self):
        with self.assertRaises(AttributeError):
            print ops.CustomDummy
        plugin_manager.load_library("./dali/test/plugins/dummy/libcustomdummyplugin.so")
        print ops.CustomDummy

    def test_pipeline_including_custom_plugin(self):
        plugin_manager.load_library("./dali/test/plugins/dummy/libcustomdummyplugin.so")
        pipe = CustomPipeline(batch_size, 1, 0)
        pipe.build()
        pipe_out = pipe.run()
        print pipe_out
        images, labels = pipe_out
        assert len(images) == batch_size
        assert len(labels) == batch_size

if __name__ == '__main__':
    unittest.main()
