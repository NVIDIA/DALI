import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module

def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = {}
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)

    if flat_dist_call.warn_on_half:
        if torch.cuda.HalfTensor in buckets:
            print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                  " It is recommended to use the NCCL backend in this case.")
            flat_dist_call.warn_on_half = False

    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        coalesced /= dist.get_world_size()
        for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module
        param_list = [param for param in self.module.state_dict().values() if torch.is_tensor(param)]
        if dist._backend == dist.dist_backend.NCCL:
            for param in param_list:
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        #broadcast parameters
        flat_dist_call(param_list, dist.broadcast, (0,) )

        #all reduce gradient hook
        def allreduce_params():
            if(self.needs_reduction):
                self.needs_reduction = False
            else:
                return
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)

        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                param._execution_engine.queue_callback(allreduce_params)
            if param.requires_grad:
                param.register_hook(allreduce_hook)


    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)
