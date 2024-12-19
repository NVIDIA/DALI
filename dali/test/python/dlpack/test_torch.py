import nvidia.dali as dali
import nvidia.dali.fn as fn
import torch
import numpy as np
from nose_utils import attr


@dali.pipeline_def(batch_size=4, num_threads=1, device_id=0, prefetch_queue_depth=2)
def _test_pipe():
    """This pipeline produces a lot of data in a very short time."""
    row = dali.types.Constant(np.full((1, 8000), 42, dtype=np.float32), device="gpu")
    col = dali.types.Constant(np.full((8000, 1), 12, dtype=np.float32), device="gpu")
    ext = fn.external_source(lambda si: np.float32(si.idx_in_epoch), batch=False, device="gpu")
    return ext + row + col


@attr("pytorch")
def test_dlpack_is_zero_copy():
    # get a DALI pipeline that produces batches of very large tensors
    pipe = _test_pipe(batch_size=1, exec_dynamic=True)

    s = torch.cuda.Stream(0)
    with torch.cuda.stream(s):
        (out,) = pipe.run(s)
        t0 = torch.from_dlpack(out[0])
        t1 = torch.from_dlpack(out[0])
        assert t0[0, 0] == 54
        assert t1[0, 0] == 54
        t0[0, 0] = 12345
        assert t1[0, 0] == 12345, "t1 and t0 should point to the same memory"


@attr("pytorch")
def test_dlpack_no_corruption():
    # get a DALI pipeline that produces batches of very large tensors
    pipe = _test_pipe(exec_dynamic=True)
    pipe.run()

    s = torch.cuda.Stream(0)
    with torch.cuda.stream(s):
        niter = 5
        means = torch.zeros((niter * pipe.max_batch_size,), dtype=torch.float32).cuda()
        flat_idx = 0
        for i in range(niter):
            (out,) = pipe.run(s)
            # convert the tensors in the batch to DLPack
            batch = [torch.from_dlpack(t) for t in out]
            for t in batch:
                means[flat_idx] = torch.mean(t)
                flat_idx += 1
        # those are meant to overwrite the results if synchronization fails
        pipe.run(s)
        pipe.run(s)
        pipe.run(s)
        pipe.run(s)
        means_cpu = means.cpu()
        for i in range(means_cpu.shape[0]):
            expected = 42 + 12 + pipe.max_batch_size + i
            assert means_cpu[i] == expected, f"{means_cpu[i]} != {expected}"
