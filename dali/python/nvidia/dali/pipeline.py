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
from nvidia.dali import tensors as Tensors
from nvidia.dali import types
from nvidia.dali import check_edge as Edge
from threading import local as tls
import warnings
pipeline_tls = tls()

def _show_deprecation_warning(deprecated, in_favor_of):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn("{} is deprecated, please use {} instead".format(deprecated, in_favor_of),
                      Warning, stacklevel=2)


class Pipeline(object):
    """Pipeline class encapsulates all data required to define and run
    DALI input pipeline.

    Parameters
    ----------
    `batch_size` : int, optional, default = -1
        Batch size of the pipeline. Negative values for this parameter
        are invalid - the default value may only be used with
        serialized pipeline (the value stored in serialized pipeline
        is used instead).
    `num_threads` : int, optional, default = -1
        Number of CPU threads used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `device_id` : int, optional, default = -1
        Id of GPU used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `seed` : int, optional, default = -1
        Seed used for random number generation. Leaving the default value
        for this parameter results in random seed.
    `exec_pipelined` : bool, optional, default = True
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
        Depth of the executor pipeline. Deeper pipeline makes DALI
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:
        ``{ "cpu_size": x, "gpu_size": y }``
        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages. It is not supported when both `exec_async`
        and `exec_pipelined` are set to `False`.
        Executor will buffer cpu and gpu stages separatelly,
        and will fill the buffer queues when the first :meth:`nvidia.dali.pipeline.Pipeline.run`
        is issued.
    `exec_async` : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`nvidia.dali.pipeline.Pipeline.run` method
        run asynchronously with respect to the calling Python thread.
        In order to synchronize with the pipeline one needs to call
        :meth:`nvidia.dali.pipeline.Pipeline.outputs` method.
    `bytes_per_sample` : int, optional, default = 0
        A hint for DALI for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    `max_streams` : int, optional, default = -1
        Limit the number of CUDA streams used by the executor.
        Value of -1 does not impose a limit.
        This parameter is currently unused (and behavior of
        unrestricted number of streams is assumed).
    `default_cuda_stream_priority` : int, optional, default = 0
        CUDA stream priority used by DALI. See `cudaStreamCreateWithPriority` in CUDA documentation
    """
    def __init__(self, batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 set_affinity=False, max_streams=-1, default_cuda_stream_priority = 0):
        self._sinks = []
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._built = False
        self._first_iter = True
        self._last_iter = False
        self._batches_to_consume = 0
        self._cpu_batches_to_consume = 0
        self._gpu_batches_to_consume = 0
        self._prepared = False
        self._names_and_devices = None
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._set_affinity = set_affinity
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._api_type = None
        self._skip_api_check = False
        if type(prefetch_queue_depth) is dict:
            self._exec_separated = True
            self._cpu_queue_size = prefetch_queue_depth["cpu_size"]
            self._gpu_queue_size = prefetch_queue_depth["gpu_size"]
            self._prefetch_queue_depth = self._cpu_queue_size  # dummy value, that will be ignored
        elif type(prefetch_queue_depth) is int:
            self._exec_separated = False
            self._prefetch_queue_depth = prefetch_queue_depth
            self._cpu_queue_size = prefetch_queue_depth
            self._gpu_queue_size = prefetch_queue_depth
        else:
            raise TypeError("Expected prefetch_queue_depth to be either int or Dict[int, int]")

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size

    @property
    def num_threads(self):
        """Number of CPU threads used by the pipeline."""
        return self._num_threads

    @property
    def device_id(self):
        """Id of the GPU used by the pipeline."""
        return self._device_id

    @property
    def exec_pipelined(self):
        return self._exec_pipelined

    @property
    def exec_async(self):
        return self._exec_async

    def epoch_size(self, name = None):
        """Epoch size of a pipeline.

        If the `name` parameter is `None`, returns a dictionary of pairs
        `(reader name, epoch size for that reader)`.
        If the `name` parameter is not `None`, returns epoch size for that
        reader.

        Parameters
        ----------
        name : str, optional, default = None
            The reader which should be used to obtain epoch size.
        """

        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if name is not None:
            return self._pipe.epoch_size(name)
        return self._pipe.epoch_size()

    @staticmethod
    def current(raise_error_if_none = True):
        pipeline = getattr(pipeline_tls, 'current_pipeline', None)
        if raise_error_if_none and (pipeline is None):
            raise RuntimeError("Unknown pipeline! "
                               "Graph edges must be created from within `define_graph` "
                               "or Pipeline.set_current() must be explicitly used.")
        return pipeline

    @staticmethod
    def set_current(pipeline):
        prev = Pipeline.current(False)
        pipeline_tls.current_pipeline = pipeline
        return prev

    def add_sink(self, edge):
        """Allows to manual add of graph edges to the pipeline which are not connected to the output and all pruned
        """
        self._sinks.append(edge)

    def _set_api_type(self, type):
        if not type in types.PipelineAPIType:
            raise RuntimeError("Wrong pipeline API set!"
                               "check available values in :meth:`nvidia.dali.types.PipelineAPIType`")
        self._api_type = type

    def _check_api_type(self, type):
        if self._api_type == None:
            self._set_api_type(type)
        if type != self._api_type:
            raise RuntimeError("Mixing pipeline API type. Currently used: " + str(self._api_type) +
                          ", but trying to use: " + str(type))

    def enable_api_check(self, enable):
        """Allows to enable or disable API check in the runtime
        """
        self._skip_api_check = not enable

    def _check_api_type_scope(self, type):
        """Checks the API currently used by pipeline and throws an error if it differs

        It helps preventing of mixing simple, iterator and scheduled based API for
        pipeline run. Disables further checks in its scope
        """
        if not self._skip_api_check:
            self._check_api_type(type)

        class api_checker():
            def __init__(self, pipe):
                self._pipe = pipe

            def __enter__(self):
                self._old_skip_api_check = self._pipe._skip_api_check
                self._pipe._skip_api_check = True

            def __exit__(self, type, value, traceback):
                self._pipe._skip_api_check = self._old_skip_api_check

        return api_checker(self)

    # Graph is constructed by backtracking from the output edges and the edges marked as sinks
    def _prepare_graph(self):
        self._pipe = b.Pipeline(self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._seed,
                                self._exec_pipelined,
                                self._prefetch_queue_depth,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams,
                                self._default_cuda_stream_priority)
        self._pipe.SetExecutionTypes(self._exec_pipelined, self._exec_separated, self._exec_async)
        self._pipe.SetQueueSizes(self._cpu_queue_size, self._gpu_queue_size)
        prev_pipeline = Pipeline.set_current(self)
        outputs = self.define_graph()
        Pipeline.set_current(prev_pipeline)
        if (not isinstance(outputs, tuple) and
            not isinstance(outputs, list)):
            outputs = (outputs,)

        for output in outputs:
            Edge._validate_edge_reference(output)

        # Backtrack to construct the graph
        op_ids = set()
        edges = deque(list(outputs) + self._sinks)
        ops = []
        while edges:
            current_edge = edges.popleft()
            source_op = current_edge.source
            if source_op is None:
                raise RuntimeError(
                    "Pipeline encountered "
                    "Edge with no source op.")

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
            for edge in source_op.inputs:
                if isinstance(edge, list):
                    for e in edge:
                        edges.append(e)
                else:
                    edges.append(edge)

        # Add the ops to the graph and build the backend
        related_logical_id = {}
        while ops:
            op = ops.pop()
            if op.relation_id not in related_logical_id:
                related_logical_id[op.relation_id] = self._pipe.AddOperator(op.spec, op.name)
            else:
                self._pipe.AddOperator(op.spec, op.name, related_logical_id[op.relation_id])
        self._prepared = True
        self._names_and_devices = [(e.name, e.device) for e in outputs]

    def build(self):
        """Build the pipeline.

        Pipeline needs to be built in order to run it standalone.
        Framework-specific plugins handle this step automatically.
        """
        if self._built:
            return

        if not self._prepared:
            self._prepare_graph()

        self._pipe.Build(self._names_and_devices)
        self._built = True

    def feed_input(self, ref, data, layout=""):
        """Bind the NumPy array to a tensor produced by ExternalSource
        operator. It is worth mentioning that `ref` **should not** be overridden
        with other operator outputs and it should be called from the
        inside of `iter_setup` method"""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        Edge._validate_edge_reference(ref)
        if isinstance(data, list):
            if self._batch_size != len(data):
                raise RuntimeError("Data list provided to feed_input needs to have batch_size length")
            inputs = []
            for datum in data:
                inputs.append(Tensors.TensorCPU(datum, layout))
            self._pipe.SetExternalTensorInput(ref.name, inputs)
        else:
            inp = Tensors.TensorListCPU(data, layout)
            self._pipe.SetExternalTLInput(ref.name, inp)

    def _run_cpu(self):
        """Run CPU portion of the pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if not self._last_iter:
            self._pipe.RunCPU()
            self._cpu_batches_to_consume += 1

    def _run_gpu(self):
        """Run GPU portion of the pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if self._cpu_batches_to_consume > 0:
            self._pipe.RunGPU()
            self._cpu_batches_to_consume -= 1
            self._gpu_batches_to_consume += 1

    def outputs(self):
        """Returns the outputs of the pipeline and releases previous buffer.

        If the pipeline is executed asynchronously, this function blocks
        until the results become available. It rises StopIteration if data set
        reached its end - usually when iter_setup cannot produce any more data"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._batches_to_consume == 0 or self._gpu_batches_to_consume == 0:
                raise StopIteration
            self._batches_to_consume -= 1
            self._gpu_batches_to_consume -= 1
            return self._outputs()

    def schedule_run(self):
        """Run the pipeline without returning the resulting buffers.

        If the pipeline was created with `exec_pipelined` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution. It provides better control to the users about when they
        want to run the pipeline, when they want to obtain resulting buffers
        and return them to DALI buffer pool when the results have been consumed.
        Needs to be used together with :meth:`nvidia.dali.pipeline.Pipeline.release_outputs`
        and :meth:`nvidia.dali.pipeline.Pipeline.share_outputs`.
        Should not be mixed with :meth:`nvidia.dali.pipeline.Pipeline.run` in the same pipeline"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._first_iter and self._exec_pipelined:
                self._prefetch()
            else:
                self._run_once()

    # for the backward compatibility
    def _run(self):
        """Deprecated. Use `nvidia.dali.pipeline.Pipeline.schedule_run` instead."""
        _show_deprecation_warning("_run", "schedule_run")
        self.schedule_run()

    def share_outputs(self):
        """Returns the outputs of the pipeline.

        Main difference to :meth:`nvidia.dali.pipeline.Pipeline.outputs`
        is that share_outputs doesn't release returned buffers, release_outputs
        need to be called for that. If the pipeline is executed asynchronously,
        this function blocks until the results become available. It provides
        the user with better control about when he wants to run the pipeline, when he wants
        to obtain the resulting buffers and when they can be returned to DALI pool when the
        results have been consumed.
        Needs to be used together with :meth:`nvidia.dali.pipeline.Pipeline.release_outputs`
        and :meth:`nvidia.dali.pipeline.Pipeline.schedule_run`
        Should not be mixed with :meth:`nvidia.dali.pipeline.Pipeline.run` in the same pipeline"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._batches_to_consume == 0 or self._gpu_batches_to_consume == 0:
                raise StopIteration
            self._batches_to_consume -= 1
            self._gpu_batches_to_consume -= 1
            return self._pipe.ShareOutputs()

    # for the backward compatibility
    def _share_outputs(self):
        """Deprecated. Use :meth:`nvidia.dali.pipeline.Pipeline.share_outputs` instead"""
        _show_deprecation_warning("_share_outputs", "share_outputs")
        self.share_outputs()

    def release_outputs(self):
        """Release buffers returned by share_outputs calls.

        It helps in case when output call result is consumed (copied)
        and buffers can be marked as free before the next call to share_outputs. It provides
        the user with better control about when he wants to run the pipeline, when he wants
        to obtain the resulting buffers and when they can be returned to DALI pool when the
        results have been consumed.
        Needs to be used together with :meth:`nvidia.dali.pipeline.Pipeline.schedule_run`
        and :meth:`nvidia.dali.pipeline.Pipeline.share_outputs`
        Should not be mixed with :meth:`nvidia.dali.pipeline.Pipeline.run` in the same pipeline"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if not self._built:
                raise RuntimeError("Pipeline must be built first.")
            return self._pipe.ReleaseOutputs()

    # for the backward compatibility
    def _release_outputs(self):
        """Deprecated. Use :meth:`nvidia.dali.pipeline.Pipeline.release_outputs` instead"""
        _show_deprecation_warning("_release_outputs", "release_outputs")
        self.release_outputs()

    def _outputs(self):
        """Release buffers previously returned and returns  the calls.

        Calling this function is equivalent to calling release_outputs
        then calling share_outputs"""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        return self._pipe.Outputs()
    
    def run(self):
        """Run the pipeline and return the result.

        If the pipeline was created with `exec_pipelined` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution.
        Should not be mixed with :meth:`nvidia.dali.pipeline.Pipeline.schedule_run` in the same pipeline,
        :meth:`nvidia.dali.pipeline.Pipeline.share_outputs` and
        :meth:`nvidia.dali.pipeline.Pipeline.release_outputs`"""
        with self._check_api_type_scope(types.PipelineAPIType.BASIC):
            self.schedule_run()
            return self.outputs()

    def _prefetch(self):
        """Executes pipeline to fill executor's pipeline."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if self._exec_separated:
            self._fill_separated_queues()
        else:
            for _ in range(self._prefetch_queue_depth):
                self._run_once()
        self._first_iter = False


    def _run_once(self):
        """Start running the whole pipeline once without waiting for its results.

        If the pipeline was created with `exec_async` option set to `True`,
        this function will return without waiting for the execution to end."""
        try:
            if not self._last_iter:
                self.iter_setup()
                self._batches_to_consume += 1
            # Special case to prevent a deadlock if user didn't release the only buffer
            if not self._exec_async and self._prefetch_queue_depth == 1:
                self.release_outputs()
            self._run_cpu()
            self._run_gpu()
        except StopIteration:
            self._last_iter = True

    def _run_up_to(self, stage_name):
        """Call the `_run_X` up to `stage_name` (inclusive).
        """
        try:
            if not self._last_iter:
                self.iter_setup()
                self._batches_to_consume += 1
                self._run_cpu()
                if stage_name == "cpu":
                    return
                self._run_gpu()
                if stage_name == "gpu":
                    return
        except StopIteration:
            self._last_iter = True


    def _fill_separated_queues(self):
        """When using separated execution fill each of the prefetch queues
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if not self._first_iter:
            raise RuntimeError("Queues can be filled only on first iteration.")
        if not self._exec_separated:
            raise RuntimeError("This function should be only used with separated execution.")
        for i in range(self._gpu_queue_size):
            self._run_up_to("gpu")
        for i in range(self._cpu_queue_size):
            self._run_up_to("cpu")

    def reset(self):
        """Resets pipeline iterator

        If pipeline iterator reached the end then reset its state to the beginning.
        """
        if self._last_iter:
            self._first_iter = True
            self._last_iter = False

    def empty(self):
        """If there is any work scheduled in the pipeline but not yet consumed
        """
        return self._batches_to_consume == 0

    def serialize(self):
        """Serialize the pipeline to a Protobuf string."""
        if not self._prepared:
            self._prepare_graph()
            self._pipe.SetOutputNames(self._names_and_devices)
        return self._pipe.SerializeToProtobuf()

    def deserialize_and_build(self, serialized_pipeline):
        """Deserialize and build the pipeline given in serialized form.

        Parameters
        ----------
        serialized_pipeline : str
                              Serialized pipeline.
        """
        self._pipe = b.Pipeline(serialized_pipeline,
                                self._batch_size,
                                self._num_threads,
                                self._device_id,
                                self._exec_pipelined,
                                self._prefetch_queue_depth,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams,
                                self._default_cuda_stream_priority)
        self._pipe.SetExecutionTypes(self._exec_pipelined, self._exec_separated, self._exec_async)
        self._pipe.SetQueueSizes(self._cpu_queue_size, self._gpu_queue_size)
        self._prepared = True
        self._pipe.Build()
        self._built = True

    def save_graph_to_dot_file(self, filename, show_tensors = False, show_ids = False,
                               use_colors = False):
        """Saves the pipeline graph to a file.

        Parameters
        ----------
        filename : str
                   Name of the file to which the graph is written.
        show_tensors : bool
                   Show the Tensor nodes in the graph (by default only Operator nodes are shown)
        show_ids : bool
                   Add the node id to the graph representation
        use_colors : bool
                   Whether use color to distinguish stages
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        self._pipe.SaveGraphToDotFile(filename, show_tensors, show_ids, use_colors)

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.

        It returns a list of outputs created by calling DALI Operators."""
        raise NotImplementedError

    def iter_setup(self):
        """This function can be overriden by user-defined
        pipeline to perform any needed setup for each iteration.
        For example, one can use this function to feed the input
        data from NumPy arrays."""
        pass
