# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import jax
import jax.numpy
import jax.dlpack

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import nvidia.dali.plugin.jax as dax


def test_tensor_passing():
    @pipeline_def(num_threads=4, batch_size=1)
    def dali_pipeline(value):
        values = fn.constant(idata=value, shape=[10], dtype=types.FLOAT, device="gpu")

        return values

    pipe = dali_pipeline(value=1, device_id=0)
    pipe.build()
    dali_data = pipe.run()

    print(dali_data)

    jax_data = dax.to_jax_array(dali_data[0].as_tensor())

    print(jax_data)
    print(jax_data.dtype)
    print(jax_data.device())

    new_data = jax_data.copy()
    new_data.at[0].set(10)

    print(new_data)
    print(jax_data)

    assert jax.numpy.array_equal(jax_data, jax.numpy.full((1, 10), 1.0))
