# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

def test_to_snake_case_impl():
    fn_name_tests = [
        ('Test', 'test'),
        ('OneTwo', 'one_two'),
        ('TestXYZ', 'test_xyz'),
        ('testX', 'test_x'),
        ('TestXx', 'test_xx'),
        ('testXX', 'test_xx'),
        ('OneXYZTwo', 'one_xyz_two'),
        ('MFCC', 'mfcc'),
        ('RandomBBoxCrop', 'random_bbox_crop'),
        ('STFT_CPU', 'stft_cpu'),
        ('DOUBLE__UNDERSCORE', 'double__underscore'),
        ('double__underscore', 'double__underscore'),
        ('XYZ1ABC', 'xyz1abc'),
        ('XYZ1abc', 'xyz1abc'),
        ('trailing__', 'trailing__'),
        ('TRAILING__', 'trailing__')
    ]

    for inp, out in fn_name_tests:
        assert fn._to_snake_case(inp) == out, f"{fn._to_snake_case(inp)} != {out}"

