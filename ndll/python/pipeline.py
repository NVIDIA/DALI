from collections import deque
import ndll.backend as b
import ndll.tensor as nt

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
        
        outputs = self.define_graph()
        if type(outputs) is not tuple:
            outputs = (outputs,)

        for t in outputs:
            if type(t) is not nt.TensorReference:
                raise TypeError(
                    "Expected outputs of type "
                    "TensorReference. Received "
                    "output type {}"
                    .format(type(t).__name__)
                )

        # Backtrack to construct the graph
        op_ids = set()
        tensors = deque(outputs)
        ops = []
        while len(tensors):
            current_tensor = tensors.popleft()
            source_op = current_tensor.source
            if source_op is None:
                raise RuntimeError(
                    "Pipeline encountered "
                    "Tensor with no source op.")

            # To make sure we don't double count ops in
            # the case that they produce more than one
            # output, we keep track of the unique op ids
            # for each op we encounter and only add the
            # op if we have not already
            if source_op.id not in op_ids:
                op_ids.add(source_op.id)
                ops.append(source_op)
                for tensor in source_op.inputs:
                    tensors.append(tensor)

        # Add the ops to the backend graph
        while len(ops):
            self._pipe.AddOperator(ops.pop().spec)
        self._built = True

    def feed_input(self, ref, data):
        if type(ref) is not nt.TensorReference:
                raise TypeError(
                    "Expected argument one to "
                    "be TensorReference. "
                    "Received output type {}"
                    .format(type(ref).__name__)
                )

        if type(data) is list:
            t = []
            for datum in data:
                t.append(nt.TensorCPU(datum))
            self._pipe.SetExternalInput(ref.name, t)
        else:
            t = nt.TensorListCPU(data)
            self._pipe.SetExternalInput(ref.name, t)
        
    # defined by the user to construct their graph of operations.
    # this returns a list of output TensorReferences that we can
    # trace back to add them to the graph
    def build_graph(self):
        raise NotImplementedError
    
    # Can be overriden by user-defined pipeline to perform any
    # needed setup for each iteration, e.g. feed in input data
    def iter_setup(self):
        pass
