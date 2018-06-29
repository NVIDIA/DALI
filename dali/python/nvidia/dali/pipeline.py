# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#pylint: disable=no-member
from collections import deque
from nvidia.dali import backend as b
from nvidia.dali import tensor as nt

class Pipeline(object):
    def __init__(self, batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                 exec_pipelined=True, exec_async=True,
                 bytes_per_sample=0, set_affinity=False,
                 max_streams=-1):
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._built = False
        self._first_iter = True
        self._prepared = False
        self._names_and_devices = None
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._set_affinity = set_affinity
        self._max_streams = max_streams

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_threads(self):
        return self._num_threads

    @property
    def device_id(self):
        return self._device_id

    def epoch_size(self, name = None):
        if not self._built:
            raise RuntimeError("Pipeline must be builti first.")
        if name is not None:
            return self._pipe.epoch_size(name)
        return self._pipe.epoch_size()

    def _prepare_graph(self):
        self._pipe = b.Pipeline(self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._seed,
                                self._exec_pipelined,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams)
        outputs = self.define_graph()
        if (not isinstance(outputs, tuple) and
            not isinstance(outputs, list)):
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
                source_op.check_args()
                ops.append(source_op)
            else:
                # If the op was already added, we need to
                # change its position to the top of the list.
                # This ensures topological ordering of ops
                # when adding to the backend pipeline
                ops.remove(source_op)
                ops.append(source_op)
            for tensor in source_op.inputs:
                if isinstance(tensor, list):
                    for t in tensor:
                        tensors.append(t)
                else:
                    tensors.append(tensor)

        # Add the ops to the graph and build the backend
        while ops:
            op = ops.pop()
            self._pipe.AddOperator(op.spec, op.name)
        self._prepared = True
        self._names_and_devices = [(t.name, t.device) for t in outputs]

    def build(self):
        if self._built:
            return

        if not self._prepared:
            self._prepare_graph()

        self._pipe.Build(self._names_and_devices)
        self._built = True

    def feed_input(self, ref, data):
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
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
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.RunCPU()

    def run_gpu(self):
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.RunGPU()

    def outputs(self):
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        return self._pipe.Outputs()

    def run(self):
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if self._first_iter and self._exec_pipelined:
            self.iter_setup()
            self.run_cpu()
            self.run_gpu()
            self._first_iter = False
        self.iter_setup()
        self.run_cpu()
        self.run_gpu()
        return self.outputs()

    def serialize(self):
        if not self._prepared:
            self._prepare_graph()
            self._pipe.SetOutputNames(self._names_and_devices)
        return self._pipe.SerializeToProtobuf()

    def deserialize_and_build(self, serialized_pipeline):
        self._pipe = b.Pipeline(serialized_pipeline,
                                self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._exec_pipelined,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams)
        self._prepared = True
        self._pipe.Build()
        self._built = True

    def save_graph_to_dot_file(self, filename):
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.SaveGraphToDotFile(filename)

    # defined by the user to construct their graph of operations.
    # this returns a list of output TensorReferences that we can
    # trace back to add them to the graph
    def define_graph(self):
        raise NotImplementedError

    # Can be overriden by user-defined pipeline to perform any
    # needed setup for each iteration, e.g. feed in input data
    def iter_setup(self):
        pass
