# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import sys
import matplotlib.pyplot as plt
from nvidia.dali import pipeline_def, fn, types

def get_data():
    s1 = np.array([2.3, 4.5, 1000.2, 4.8, 6.8, 4.5], dtype=np.float32)
    s2 = np.array([5.53, 4.6, 10.2, 0.8, 0.3], dtype=np.float32)
    s3 = [5.3, 94.6, 10.2, 0.8, 0.3]
    s4 = [5.23, 4.6, 10.2, 0.85, 0.3, 8.9, 2.3]
    s5 = [5.3, 4.6, 103.2, 0.8, 0.36, 4.4]

    return [s1]

@pipeline_def(num_threads = 1, device_id = 0)
def get_pipeline():
    data = fn.external_source(get_data, batch=True, dtype=types.FLOAT)
    result = fn.cwt(data.gpu(), device="gpu", a=[1.0, 2.0, 4.5], wavelet=types.MEXH, wavelet_args=[1.0])
    return data, result

pipe = get_pipeline(batch_size=1, device_id=0)
pipe.build()
data, result = pipe.run()
print(result.as_cpu())
res = [np.array(r) for r in result.as_cpu()]

np.set_printoptions(threshold=sys.maxsize)
print(res[0][0][0])