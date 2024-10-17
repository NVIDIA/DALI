import nvidia.dali as dali
import nvidia.dali.fn as fn
import torch
import numpy as np
import time

@dali.pipeline_def(batch_size=32, num_threads=8, device_id=0)
def test_pipe():
    row = dali.types.Constant(np.full((1, 1000), 42, dtype=np.float32), device="gpu")
    col = dali.types.Constant(np.full((10000, 1), 12, dtype=np.float32), device="gpu")
    ext = fn.external_source(lambda si: np.float32(si.idx_in_epoch), batch=False, device="gpu")
    return ext + row + col


def test_dlpack():
    print("Testing dlpack")
    pipe = test_pipe(experimental_exec_dynamic=True)
    pipe.build()
    pipe.run()

    results = []
    s = torch.cuda.Stream(0)
    with torch.cuda.stream(s):
        sum = torch.zeros((), dtype=torch.float32).cuda()
        sum -= 318
        start = time.time_ns()
        for i in range(10):
            out, = pipe.run(s)
            batch = [torch.from_dlpack(t) for t in out]
            sum += torch.mean(torch.stack(batch))
        sum_cpu = sum.cpu()
        end = time.time_ns()
        print((end-start)*1e-6)
        print(sum_cpu)
    return (end-start)*1e-6


def test_copy(new_exec):
    print(f"Testing output copy with {'new' if new_exec else 'old'} executor")
    pipe = test_pipe(experimental_exec_dynamic=new_exec)
    pipe.build()
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

    results = []
    s = torch.cuda.Stream(0)
    with torch.cuda.stream(s):
        sum = torch.zeros((), dtype=torch.float32).cuda()
        sum -= 318
        start = time.time_ns()
        for i in range(10):
            out, = pipe.run()
            batch = [to_torch(t) for t in out]
            sum += torch.mean(torch.stack(batch))
        sum_cpu = sum.cpu()
        end = time.time_ns()
        print(sum_cpu)
        print((end-start)*1e-6)
    pipe._shutdown()
    del pipe
    del sum
    del sum_cpu
    return (end-start)*1e-6

dlpack_times = []
copy_new_times = []
copy_old_times = []

for i in range(30):
    dlpack_times.append(test_dlpack())
    copy_new_times.append(test_copy(True))
    copy_old_times.append(test_copy(False))

print("Times")
print("\tBest\tMedian\tMean\tMax")
print(f"DLPack\t{np.min(dlpack_times)}\t{np.median(dlpack_times)}\t{np.mean(dlpack_times)}\t{np.max(dlpack_times)}")
print(f"Copy (new)\t{np.min(copy_new_times)}\t{np.median(copy_new_times)}\t{np.mean(copy_new_times)}\t{np.max(copy_new_times)}")
print(f"Copy (old)\t{np.min(copy_old_times)}\t{np.median(copy_old_times)}\t{np.mean(copy_old_times)}\t{np.max(copy_old_times)}")

