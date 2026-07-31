# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    resize2_func = ndd._op_builder.build_fn_wrapper(Resize2, "resize2", False)
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

    the_op = _conversion_test_op(check_converted)
    # keep it as a variable - the type is mutable, so we don't treat it as invariant and
    # therefore it's not exempt from conversion
    size = [100, 100]
    x = the_op(img, size=size)
    x.evaluate()
    assert test_calls == 1, "Argument check function not called"

    the_op = _conversion_test_op(check_converted)
    size = ndd.tensor([100, 100])
    x = the_op(img, size=size)
    x.evaluate()
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

    test_calls_1 = 0
    test_calls_2 = 0

    def check_converted(args):
        nonlocal test_calls_1
        test_calls_1 += 1
        assert args["size"].dtype == ndd.float32, "size should be float32"

    the_op = _conversion_test_op(check_converted)
    size = [100, 100]
    # keep it as a variable - the type is mutable, so we don't treat it as invariant and
    # therefore it's not exempt from conversion
    x = the_op(imgs, size=size)
    x.evaluate()
    assert test_calls_1 == 1, "Argument check function not called"
    size = ndd.batch([[100, 100], [150, 150]])
    the_op = _conversion_test_op(check_converted)
    x = the_op(imgs, size=size)
    x.evaluate()
    assert test_calls_1 == 2, "Argument check function not called"

    size = ndd.batch([[100, 100], [150, 150]], dtype=ndd.float32)

    def check_not_converted(args):
        nonlocal test_calls_2
        test_calls_2 += 1
        assert args["size"]._storage is size._storage, "size should be passed as-is"

    the_op = _conversion_test_op(check_not_converted)
    x = the_op(imgs, size=size)
    x.evaluate()
    assert test_calls_2 == 1, "Argument check function not called"
