# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

def test_element_extract_operator():
    batch_size = 4
    F = 10
    W = 32
    H = 32
    C = 3

    test_data = []
    for _ in range(batch_size):
        test_data.append( np.array( np.random.rand(F, H, W, C) * 255, dtype = np.uint8 ) )

    class ExternalInputIterator(object):
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def __iter__(self):
            self.i = 0
            self.n = self.batch_size
            return self

        def __next__(self):
            batch = test_data
            self.i = (self.i + 1) % self.n
            return (batch)

        next = __next__

    eii = ExternalInputIterator(batch_size)
    iterator = iter(eii)

    class ElementExtractPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(ElementExtractPipeline, self).__init__(batch_size, num_threads, device_id)
            self.inputs = ops.ExternalSource()
            # Extract first element in each sample
            self.element_extract_first = ops.ElementExtract(element_map=[0])
            # Extract last element in each sample
            self.element_extract_last = ops.ElementExtract(element_map=[F-1])
            # Extract both first and last element in each sample to two separate outputs
            self.element_extract_first_last = ops.ElementExtract(element_map=[0, F-1])

        def define_graph(self):
            self.sequences = self.inputs()
            first_element_1 = self.element_extract_first(self.sequences)
            last_element_1 = self.element_extract_last(self.sequences)
            first_element_2, last_element_2 = self.element_extract_first_last(self.sequences)
            return (first_element_1, last_element_1, first_element_2, last_element_2)

        def iter_setup(self):
            sequences = iterator.next()
            self.feed_input(self.sequences, sequences)


    pipe = ElementExtractPipeline(batch_size, 1, 0)
    pipe.build()
    pipe_out = pipe.run()
    output1, output2, output3, output4 = pipe_out

    assert len(output1) == batch_size
    assert len(output2) == batch_size
    assert len(output3) == batch_size
    assert len(output4) == batch_size

    for i in range(batch_size):
        out1 = output1.at(i)
        out2 = output2.at(i)
        out3 = output3.at(i)
        out4 = output4.at(i)

        expected_first = test_data[i][0]
        assert out1.shape == out3.shape
        np.testing.assert_array_equal( expected_first, out1 )
        np.testing.assert_array_equal( expected_first, out3 )

        expected_last = test_data[i][F-1]
        assert out2.shape == out4.shape
        np.testing.assert_array_equal( expected_last, out2 )
        np.testing.assert_array_equal( expected_last, out4 )
