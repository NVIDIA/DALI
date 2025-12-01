# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def, Pipeline

import numpy as np
from nose_utils import raises


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def pipeline():
    output = fn.external_source(source=np.zeros((8, 8)), name="input")
    return output


@raises(
    RuntimeError,
    glob="Cannot use `feed_input` on the external source 'input' with a `source`"
    " argument specified.",
)
def test_feed_input_with_source():
    pipe = pipeline()
    pipe.feed_input("input", np.zeros((8, 8)))
    pipe.run()


def test_external_source_with_callback():
    """Test if using external_source with 'source' doesn't raise exceptions."""
    pipe = pipeline()
    pipe.run()


def test_external_source_with_serialized_pipe():
    @pipeline_def
    def serialized_pipe():
        return fn.external_source(name="es")

    pipe = serialized_pipe(batch_size=10, num_threads=3, device_id=0)
    serialized_str = pipe.serialize()
    deserialized_pipe = Pipeline(10, 4, 0)
    deserialized_pipe.deserialize_and_build(serialized_str)
    deserialized_pipe.feed_input("es", np.zeros([10, 10]))
