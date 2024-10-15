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


# pipe = test_pipe(experimental_exec_dynamic=True)
# pipe.build()
# pipe.run()

# def to_torch(dali_tensor):
#     import nvidia.dali.plugin.pytorch as dali_pyt
#     device = None
#     stream = None
#     if type(dali_tensor) is dali.backend.TensorGPU:
#         device = torch.device("cuda", dali_tensor.device_id())
#         stream = torch.cuda.current_stream(device=device)
#     dtype = dali_pyt.to_torch_type[dali_tensor.dtype]
#     t = torch.empty(dali_tensor.shape(), dtype=dtype, device=device)
#     dali_pyt.feed_ndarray(dali_tensor, t, stream)
#     return t

# results = []
# s = torch.cuda.Stream(0)
# with torch.cuda.stream(s):
#     sum = torch.zeros((), dtype=torch.float32).cuda()
#     sum -= 318
#     start = time.time_ns()
#     for i in range(10):
#         out, = pipe.run()
#         batch = [to_torch(t) for t in out]
#         sum += torch.mean(torch.stack(batch))
#     print(sum)
#     end = time.time_ns()
#     print((end-start)*1e-6)
