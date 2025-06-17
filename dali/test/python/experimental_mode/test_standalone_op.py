# import nvidia.dali.experimental.dali2 as dali2
import nvidia.dali as dali
import nvidia.dali.backend_impl as _b
import numpy as np


def complete_spec(spec):
    spec.AddArg("num_threads", 4)
    spec.AddArg("max_batch_size", 1)
    spec.AddArg("device_id", dali.backend_impl.GetCUDACurrentDevice())


def test_standalone_op():
    # execution environment
    tp = _b._ThreadPool(4)
    stream = _b.Stream(0)

    # standalone operator invocation
    # mock inputs
    a = dali.data_node.DataNode("a", "cpu", None)
    b = dali.data_node.DataNode("b", "gpu", None)
    # the operation - in this case an arithmetic operation
    x = a + b
    # the spec of the operator
    spec = x.source.spec
    # complete the spec with the execution environment
    complete_spec(spec)
    print(spec)
    # create the operator
    op = _b._Operator(spec)
    # create the workspace and populat ethe environment
    ws = _b._Workspace(tp)
    ws.SetStream(stream)
    # actual inputs
    A = dali.tensors.TensorListCPU([np.int32([1, 2, 3])])._as_gpu()
    B = dali.tensors.TensorListCPU([np.int32([4, 5, 6])])._as_gpu()
    ws.AddInput(A)
    ws.AddInput(B)
    # run the operator
    op.SetupAndRun(ws)
    # get the output
    (out,) = ws.GetOutputs()
    out = np.array(out[0].as_cpu())
    assert np.array_equal(out, np.int32([5, 7, 9]))
