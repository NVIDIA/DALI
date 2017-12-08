from collections import deque
import ndll.backend as b
from ndll.tensor import TensorReference

# Python format documentation
# Python style guide
# Auto-expose ndll.ops [x]
# ndll.tensor module w/ TensorReference, TensorCPU, TensorGPU
class Pipeline(object):
    def __init__(self, batch_size, num_threads, device_id, queue_depth = 2,
                 bytes_per_sample = 0, set_affinity = False, max_streams = -1):
        # Note: We initialize NDLL with default allocators here. If
        # a framework wants to hook their allocators up through the
        # python API, we will need to stop doing this
        b.Init(b.OpSpec("PinnedCPUAllocator"), b.OpSpec("GPUAllocator"))
        self._pipe = b.Pipeline(batch_size,
                                num_threads,
                                device_id,
                                queue_depth,
                                bytes_per_sample,
                                set_affinity,
                                max_streams)
        self._built = False

    def build(self):
        if self._built:
            raise RuntimeError("build() can only be called once.")
        
        outputs = self.build_graph()
        outputs = [outputs] if type(outputs) is not list else outputs

        for tensor in outputs:
            if type(tensor) is not TensorReference:
                raise TypeError(
                    """Expected outputs of type
                    TensorReference. Received 
                    output type {}"""
                    .format(type(tensor).__name__)
                )

        # Backtrack to construct the graph
        tensors = deque(outputs)
        ops = []
        while len(tensors):
            # TODO(tgale): This will produce duplicate ops
            # if an op outputs more than one tensor
            current_tensor = tensors.popleft()
            source_op = current_tensor.source
            if source_op is None:
                raise RuntimeError(
                    """Pipeline encountered 
                    Tensor with no source op.""")
            
            ops.append(source_op)
            for tensor in source_op.inputs:
                tensors.append(tensor)

        # Add the ops to the backend graph
        while len(ops):
            self._pipe.AddOperator(ops.pop().spec)
        self._built = True
        
    # defined by the user to construct their graph of operations.
    # this returns a list of output TensorReferences that we can
    # trace back to add them to the graph
    def build_graph(self):
        raise NotImplementedError
    
