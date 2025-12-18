# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from test_utils import get_dali_extra_path


def _conversion_test_op(check_arg_func):
    class Resize2(ndd._ops.Resize):
        def _run(self, ctx, *inputs, **args):
            check_arg_func(args)
            return ndd._ops.Resize._run(self, ctx, *inputs, **args)

    resize2_func = ndd._op_builder.build_fn_wrapper(Resize2)
    return resize2_func


def test_arg_conversion():
    path = os.path.join(get_dali_extra_path(), "db", "imgproc", "alley.png")
    file = np.fromfile(path, dtype=np.uint8)
    img = ndd.decoders.image(file)

    test_calls = 0

    def check_converted(args):
        nonlocal test_calls
        test_calls += 1
        assert args["size"].dtype == ndd.float32, "size should be float32"

    _conversion_test_op(check_converted)(img, size=[100, 100]).evaluate()
    assert test_calls == 1, "Argument check function not called"
    size = ndd.tensor([100, 100])
    _conversion_test_op(check_converted)(img, size=size).evaluate()
    assert test_calls == 2, "Argument check function not called"

    size = ndd.tensor([100, 100], dtype=ndd.float32)

    def check_not_converted(args):
        nonlocal test_calls
        test_calls += 1
        assert args["size"]._storage is size._storage, "size should be passed as-is"

    _conversion_test_op(check_not_converted)(img, size=size).evaluate()
    assert test_calls == 3, "Argument check function not called"


def test_arg_conversion_batch():
    path = os.path.join(get_dali_extra_path(), "db", "imgproc", "alley.png")
    file = np.fromfile(path, dtype=np.uint8)
    img = ndd.decoders.image(file)
    imgs = ndd.as_batch([img, img])

    test_calls = 0

    def check_converted(args):
        nonlocal test_calls
        test_calls += 1
        assert args["size"].dtype == ndd.float32, "size should be float32"

    _conversion_test_op(check_converted)(imgs, size=[100, 100]).evaluate()
    assert test_calls == 1, "Argument check function not called"
    size = ndd.batch([[100, 100], [150, 150]])
    _conversion_test_op(check_converted)(imgs, size=size).evaluate()
    assert test_calls == 2, "Argument check function not called"

    size = ndd.batch([[100, 100], [150, 150]], dtype=ndd.float32)

    def check_not_converted(args):
        nonlocal test_calls
        test_calls += 1
        assert args["size"]._storage is size._storage, "size should be passed as-is"

    _conversion_test_op(check_not_converted)(imgs, size=size).evaluate()
    assert test_calls == 3, "Argument check function not called"
