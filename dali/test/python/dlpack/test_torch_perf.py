import nvidia.dali as dali
import nvidia.dali.fn as fn
import torch
import numpy as np
import time
from nose_utils import attr


@dali.pipeline_def(batch_size=32, num_threads=8, device_id=0)
def test_pipe():
    row = dali.types.Constant(np.full((1, 1000), 42, dtype=np.float32), device="gpu")
    col = dali.types.Constant(np.full((1000, 1), 12, dtype=np.float32), device="gpu")
    ext = fn.external_source(lambda si: np.float32(si.idx_in_epoch), batch=False, device="gpu")
    return ext + row + col


stream = torch.cuda.Stream(0)


def bench_dlpack(verbose=False):
    if verbose:
        print("Testing dlpack")
    pipe = test_pipe(exec_dynamic=True)
    pipe.run()

    with torch.cuda.stream(stream):
        sum = torch.zeros((), dtype=torch.float32).cuda()
        sum -= 318
        start = time.time_ns()
        for i in range(10):
            (out,) = pipe.run(stream)
            batch = [torch.from_dlpack(t) for t in out]
            batch_tensor = torch.stack(batch)
            sum += torch.mean(batch_tensor)
        sum_cpu = sum.cpu()
        assert sum_cpu == 2137, sum_cpu
        end = time.time_ns()
        if verbose:
            print((end - start) * 1e-6)
    return (end - start) * 1e-6


def bench_copy(new_exec, verbose=False):
    if verbose:
        print(f"Testing output copy with {'new' if new_exec else 'old'} executor")
    pipe = test_pipe(exec_dynamic=new_exec)
    pipe.run()

    def to_torch(dali_tensor):
        import nvidia.dali.plugin.pytorch as dali_pyt

        device = None
        stream = None
        if type(dali_tensor) is dali.backend.TensorGPU:
            device = torch.device("cuda", dali_tensor.device_id())
            stream = torch.cuda.current_stream(device=device)
        dtype = dali_pyt.to_torch_type[dali_tensor.dtype]
        t = torch.empty(dali_tensor.shape(), dtype=dtype, device=device)
        dali_pyt.feed_ndarray(dali_tensor, t, stream)
        return t

    with torch.cuda.stream(stream):
        sum = torch.zeros((), dtype=torch.float32).cuda()
        sum -= 318
        start = time.time_ns()
        for i in range(10):
            (out,) = pipe.run()
            batch = [to_torch(t) for t in out]
            sum += torch.mean(torch.stack(batch))
        sum_cpu = sum.cpu()
        end = time.time_ns()
        assert sum_cpu == 2137, sum_cpu
        if verbose:
            print((end - start) * 1e-6)
    pipe._shutdown()
    return (end - start) * 1e-6


@attr("pytorch")
def test_perf():
    """Test that DLPack zero-copy output is faster than copying."""
    dlpack_times = []
    copy_new_times = []
    copy_old_times = []

    runs = 20
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        dlpack_times.append(bench_dlpack())
        copy_old_times.append(bench_copy(False))
        copy_new_times.append(bench_copy(True))

    print("Times")
    print("\tBest\tMedian\tMean\tMax")
    print(
        f"DLPack\t{np.min(dlpack_times)}\t{np.median(dlpack_times)}"
        f"\t{np.mean(dlpack_times)}\t{np.max(dlpack_times)}"
    )
    print(
        f"Copy (new)\t{np.min(copy_new_times)}\t{np.median(copy_new_times)}"
        f"\t{np.mean(copy_new_times)}\t{np.max(copy_new_times)}"
    )
    print(
        f"Copy (old)\t{np.min(copy_old_times)}\t{np.median(copy_old_times)}"
        f"\t{np.mean(copy_old_times)}\t{np.max(copy_old_times)}"
    )
    assert np.min(dlpack_times) < np.min(copy_new_times)
    assert np.min(dlpack_times) < np.min(copy_old_times)
    assert np.median(dlpack_times) < np.median(copy_new_times)
    assert np.median(dlpack_times) < np.median(copy_old_times)
