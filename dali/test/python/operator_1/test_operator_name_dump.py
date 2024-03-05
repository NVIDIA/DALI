# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import traceback
import re
import fnmatch

from nvidia.dali import pipeline_def, fn, ops, Pipeline
from nose2.tools import params
from test_utils import load_test_operator_plugin


def setUpModule():
    load_test_operator_plugin()


def extract_str_from_tl(out):
    """Extract string from the test operator that returns it as u8 tensor."""
    # Extract data from first sample
    arr = np.array(out[0])
    # View it as list of characters and decode to string
    return arr.view(f"S{arr.shape[0]}")[0].decode("utf-8")


api_variants = [("fn", fn.name_dump), ("ops", ops.NameDump)]


@params(*api_variants)
def test_api_name(api, op):

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="api")

    p = pipe()
    p.build()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    assert out_str == "api"
