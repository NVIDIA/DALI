#pylint: disable=no-member
from collections import deque
import ndll.backend as b
import ndll.tensor as nt

class Pipeline(object):
    def __init__(self, batch_size, num_threads, device_id,
                 exec_pipelined=False, exec_async=False,
                 bytes_per_sample=0, set_affinity=False,
                 max_streams=-1):
        self._pipe = b.Pipeline(batch_size,
                                num_threads,
                                device_id,
                                exec_pipelined,
                                exec_async,
                                bytes_per_sample,
                                set_affinity,
                                max_streams)
        self._exec_pipelined = exec_pipelined
        self._built = False
        self._first_iter = True

    @property
    def batch_size(self):
        return self._pipe.batch_size()

    @property
    def num_threads(self):
        return self._pipe.num_threads()

    @property
    def device_id(self):
        return self._pipe.device_id()

    def build(self):
        if self._built:
            raise RuntimeError("build() can only be called once.")

        outputs = self.define_graph()
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for output in outputs:
            if not isinstance(output, nt.TensorReference):
                raise TypeError(
                    "Expected outputs of type "
                    "TensorReference. Received "
                    "output type {}"
                    .format(type(output).__name__)
                )

        # Backtrack to construct the graph
        op_ids = set()
        tensors = deque(outputs)
        ops = []
        while tensors:
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
            else:
                # If the op was already added, we need to
                # change its position to the top of the list.
                # This ensures topological ordering of ops
                # when adding to the backend pipeline
                ops.remove(source_op)
                ops.append(source_op)

        # Add the ops to the graph and build the backend
        while ops:
            self._pipe.AddOperator(ops.pop().spec)
        names_and_devices = [(t.name, t.device) for t in outputs]
        self._pipe.Build(names_and_devices)
        self._built = True

    def feed_input(self, ref, data):
        if not isinstance(ref, nt.TensorReference):
            raise TypeError(
                "Expected argument one to "
                "be TensorReference. "
                "Received output type {}"
                .format(type(ref).__name__)
            )
        if isinstance(data, list):
            inputs = []
            for datum in data:
                inputs.append(nt.TensorCPU(datum))
            self._pipe.SetExternalTensorInput(ref.name, inputs)
        else:
            inp = nt.TensorListCPU(data)
            self._pipe.SetExternalTLInput(ref.name, inp)

    def run_cpu(self):
        self._pipe.RunCPU()

    def run_gpu(self):
        self._pipe.RunGPU()

    def outputs(self):
        return self._pipe.Outputs()

    def run(self):
        if self._first_iter and self._exec_pipelined:
            self.iter_setup()
            self.run_cpu()
            self.run_gpu()
            self._first_iter = False
        self.iter_setup()
        self.run_cpu()
        self.run_gpu()
        return self.outputs()

    # defined by the user to construct their graph of operations.
    # this returns a list of output TensorReferences that we can
    # trace back to add them to the graph
    def build_graph(self):
        raise NotImplementedError

    # Can be overriden by user-defined pipeline to perform any
    # needed setup for each iteration, e.g. feed in input data
    def iter_setup(self):
        pass
