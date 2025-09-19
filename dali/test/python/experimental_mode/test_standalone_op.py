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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.backend_impl as _b
import numpy as np
from nose2.tools import params
import os


def complete_spec(spec, max_batch_size=1):
    spec.AddArg("num_threads", 4)
    spec.AddArg("max_batch_size", max_batch_size)
    spec.AddArg("device_id", dali.backend_impl.GetCUDACurrentDevice())


@params(("cpu", "cpu"), ("gpu", "cpu"), ("cpu", "gpu"), ("gpu", "gpu"))
def test_standalone_arithm_op(device1, device2):
    # execution environment
    tp = _b._ThreadPool(4)
    stream = _b.Stream(0)

    # standalone operator invocation
    # mock inputs
    a = dali.data_node.DataNode("a", device1, None)
    b = dali.data_node.DataNode("b", device2, None)
    # the operation - in this case an arithmetic operation
    x = a + b
    # the spec of the operator
    spec = x.source.spec
    # complete the spec with the execution environment
    complete_spec(spec)
    # create the operator
    op = _b._Operator(spec)
    # create the workspace and populat ethe environment
    ws = _b._Workspace(tp)
    ws.SetStream(stream)
    # actual inputs
    A = dali.tensors.TensorListCPU([np.int32([1, 2, 3])])
    B = dali.tensors.TensorListCPU([np.int32([4, 5, 6])])
    if device1 == "gpu" or device2 == "gpu":
        A = A._as_gpu()
        B = B._as_gpu()
    ws.AddInput(A)
    ws.AddInput(B)
    # run the operator
    op.SetupAndRun(ws)
    # get the output
    (out,) = ws.GetOutputs()
    out = np.array(out[0].as_cpu())
    assert np.array_equal(out, np.int32([5, 7, 9]))


def test_argument_input():
    # execution environment
    tp = _b._ThreadPool(4)
    stream = _b.Stream(0)

    # standalone operator invocation
    # mock inputs
    a = dali.data_node.DataNode("a", "cpu", None)
    b = dali.data_node.DataNode("b", "cpu", None)
    c = dali.data_node.DataNode("c", "cpu", None)
    # the operation - in this case an arithmetic operation
    x = fn.normalize(a, mean=b, stddev=c)
    # the spec of the operator
    spec = x.source.spec
    # complete the spec with the execution environment
    complete_spec(spec, max_batch_size=10)  # oversized max batch size
    # create the operator
    op = _b._Operator(spec)
    # create the workspace and populat ethe environment
    ws = _b._Workspace(tp)
    ws.SetStream(stream)
    # actual inputs
    A = dali.tensors.TensorListCPU(
        [
            np.int32([4, 5, 6]),
            np.int32([6, 2, -2]),
            np.int32([13, 18, 23]),
        ]
    )
    B = dali.tensors.TensorListCPU([np.float32([1]), np.float32([2]), np.float32([3])])
    C = dali.tensors.TensorListCPU([np.float32([1]), np.float32([4]), np.float32([5])])
    ws.AddInput(A)
    ws.AddArgumentInput("mean", B)
    ws.AddArgumentInput("stddev", C)
    # run the operator
    op.SetupAndRun(ws, batch_size=3)
    # get the output
    (out,) = ws.GetOutputs()
    np.array_equal(out[0], np.float32([3, 4, 5]))
    np.array_equal(out[1], np.float32([1, 0, -1]))
    np.array_equal(out[2], np.float32([2, 3, 4]))


dali_extra_path = os.environ.get("DALI_EXTRA_PATH", None)
if dali_extra_path is None:
    raise RuntimeError("DALI_EXTRA_PATH is not set")
path = os.path.join(dali_extra_path, "db", "single", "jpeg")
if not os.path.exists(path):
    raise RuntimeError(f'The path "{path}" does not exist')


def test_standalone_op_reader():
    # execution environment
    tp = _b._ThreadPool(4)
    stream = _b.Stream(0)

    batch_size = 10

    # standalone operator invocation
    reader_outs = fn.readers.file(
        file_root=path,
        file_list=os.path.join(path, "image_list.txt"),
        random_shuffle=False,
        seed=123,
    )
    # the spec of the operator
    spec = reader_outs[0].source.spec
    # complete the spec with the execution environment
    complete_spec(spec, max_batch_size=batch_size)
    # create the operator
    op = _b._Operator(spec)
    # create the workspace and populat ethe environment

    pipe = dali.Pipeline(batch_size, 4, 0)
    ref_reader_outs = fn.readers.file(
        name="reader",
        file_root=path,
        file_list=os.path.join(path, "image_list.txt"),
        random_shuffle=False,
        seed=123,
    )
    pipe.set_outputs(*ref_reader_outs)

    pipe.build()
    out_meta = op.GetReaderMeta()
    ref_meta = pipe.reader_meta("reader")
    assert out_meta == ref_meta

    for i in range(10):
        ws = _b._Workspace(tp)
        ws.SetStream(stream)
        op.SetupAndRun(ws)
        # get the output
        (out_img, out_label) = ws.GetOutputs()
        (ref_img, ref_label) = pipe.run()
        for i in range(batch_size):
            assert np.array_equal(out_img[i], ref_img[i])
            assert np.array_equal(out_label[i], ref_label[i])
