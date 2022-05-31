import torch
import numpy as np
import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

shape = [5000000]
batch_size = 8

@pipeline_def(batch_size=batch_size, device_id=0, num_threads=8)
def _test_pipe():
    get_tensor = lambda si: np.arange(si.idx_in_epoch, si.idx_in_epoch + shape[0], dtype=np.int32)
    inp = fn.external_source(get_tensor, batch=False)
    return inp.gpu()


from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import numpy as np

to_torch_type = {
    types.DALIDataType.FLOAT   : torch.float32,
    types.DALIDataType.FLOAT64 : torch.float64,
    types.DALIDataType.FLOAT16 : torch.float16,
    types.DALIDataType.UINT8   : torch.uint8,
    types.DALIDataType.INT8    : torch.int8,
    types.DALIDataType.INT16   : torch.int16,
    types.DALIDataType.INT32   : torch.int32,
    types.DALIDataType.INT64   : torch.int64
}

def feed_ndarray(dali_tensor, arr, cuda_stream = None, non_blocking = False):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, ("The element type of DALI Tensor/TensorList"
            " doesn't match the element type of the target PyTorch Tensor: {} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer, None if cuda_stream is None else ctypes.c_void_p(cuda_stream), non_blocking)
    return arr

num_iters = 20

def ref_tensor(batch_size, sample_shape, start_value):
    volume = np.prod(sample_shape)
    sample0 = torch.arange(start_value, start_value + volume, dtype=torch.int32, device="cuda:0").reshape(shape)
    return torch.stack([sample0 + i for i in range(batch_size)])

def check(arr, ref):
    return torch.equal(arr, ref)

# get a Pytorch CUDA stream
stream = torch.cuda.Stream(device=0)
with torch.cuda.stream(stream):
    # allocate an empty tensor into which the pipeline output will be copied
    arr = torch.empty([batch_size] + shape, dtype=torch.int32, device="cuda:0")
    # create a reference tensor...
    ref = ref_tensor(batch_size, shape, 0)
    # ...and tensors which will be used to hog the GPU
    hog = [ref_tensor(batch_size, shape, i * batch_size) for i in range(10)]

    # try 10 times
    for i in range(10):
        # create a fresh pipeline
        pipe = _test_pipe(prefetch_queue_depth=2)
        pipe.build()
        # schedule some runs ahead, so we know that the execution
        # of the next iteration starts immediately
        pipe.schedule_run()
        pipe.schedule_run()
        out, = pipe.share_outputs()
        # do something time-consuming on the torch stream to give DALI time to clobber the buffer
        hog = [torch.sqrt(x) for x in hog]
        # copy the result asynchronously
        feed_ndarray(out.as_tensor(), arr, stream.cuda_stream, True)
        pipe.release_outputs()
        # drain
        _, = pipe.share_outputs()
        pipe.release_outputs()
        # if no appropriate synchronization is done, the array is likely clobbered with the
        # results from the second iteration
        assert check(arr, ref[0])
        del pipe
