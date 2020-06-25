# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali.backend import CheckDLPackCapsule
from threading import local as tls
from . import data_node as _data_node
import warnings
import ctypes
pipeline_tls = tls()

from .data_node import DataNode
DataNode.__module__ = __name__      # move to pipeline

def _show_deprecation_warning(deprecated, in_favor_of):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn("{} is deprecated, please use {} instead".format(deprecated, in_favor_of),
                      Warning, stacklevel=2)

def _get_default_stream_for_array(array):
    if types._is_torch_tensor(array):
        import torch
        return torch.cuda.current_stream().cuda_stream
    elif types._is_cupy_array(array):
        import cupy
        return cupy.cuda.get_current_stream().ptr
    else:
        return None

class Pipeline(object):
    """Pipeline class is the base of all DALI data pipelines. The pipeline
encapsulates the data processing graph and the execution engine.

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
    and will fill the buffer queues when the first :meth:`run`
    is issued.
`exec_async` : bool, optional, default = True
    Whether to execute the pipeline asynchronously.
    This makes :meth:`run` method
    run asynchronously with respect to the calling Python thread.
    In order to synchronize with the pipeline one needs to call
    :meth:`outputs` method.
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
`enable_memory_stats`: bool, optional, default = False
    If DALI should print operator output buffer statistics.
    Usefull for `bytes_per_sample_hint` operator parameter.
"""
    def __init__(self, batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 set_affinity=False, max_streams=-1, default_cuda_stream_priority = 0,
                 *,
                 enable_memory_stats=False):
        self._sinks = []
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._built = False
        self._first_iter = True
        self._last_iter = False
        self._iter = 0
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
        self._graph_out = None
        self._input_callbacks = None
        self._enable_memory_stats = enable_memory_stats
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
            return self._pipe.reader_meta(name)["epoch_size_padded"]
        return {name : v["epoch_size_padded"] for k, v in self._pipe.reader_meta()}

    def executor_statistics(self):
        """Returns provided pipeline executor statistics metadata as a dictionary.
        Each key in the dictionary is the operator name. To enable it use ``executor_statistics``

        Available metadata keys for each operator:

        ``real_memory_size``:     list of memory sizes that is used by each output of the operator;
                                  index in the list corresponds to the output index

        ``max_real_memory_size``: list of maximum tensor size that is used by each output of the operator;
                                  index in the list corresponds to the output index

        ``reserved_memory_size``: list of memory sizes that is reserved for each of the operator outputs
                                  index in the list corresponds to the output index

        ``max_reserved_memory_size``: list of maximum memory sizes per tensor that is reserved for each of the operator outputs
                                  index in the list corresponds to the output index
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        return self._pipe.executor_statistics()

    def reader_meta(self, name = None):
        """Returns provided reader metadata as a dictionary. If no name is provided if provides
        a dictionary with data for all readers as {reader_name : meta}

        Available metadata keys:

        ``epoch_size``:        raw epoch size

        ``epoch_size_padded``: epoch size with the padding at the end to be divisible by the number of shards

        ``number_of_shards``:  number of shards

        ``shard_id``:          shard id of given reader

        ``pad_last_batch``:    if given reader should pad last batch

        ``stick_to_shard``:    if given reader should stick to its shard

        Parameters
        ----------
        name : str, optional, default = None
            The reader which should be used to obtain shards_number.
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if name is not None:
            return self._pipe.reader_meta(name)
        return self._pipe.reader_meta()

    @staticmethod
    def current():
        return getattr(pipeline_tls, 'current_pipeline', None)

    @staticmethod
    def _raise_pipeline_required(op_name):
        raise RuntimeError("Current Pipeline not set!\n" +
            op_name + " operator must be used inside `define_graph` or "
            "current pipeline must be explicitly set using context manager (`with my_pipeline:`) "
            "or with a call to `Pipeline.push_current(my_pipeline)`.")

    @staticmethod
    def push_current(pipeline):
        """Sets the pipeline as current and stores the previous current pipeline
        on stack. To restore previous pipeline as current, use :meth:`pop_current`.

        To make sure that the pipeline is properly restored in case of exception, use context
        manager (`with my_pipeline:`).

        Current pipeline is required to call operators with side effects or without outputs.
        Examples of such operators are `PythonFunction` (potential side effects) or `DumpImage`
        (no output).

        Any dangling operator can be marked as having side effects if it's marked
        with `preserve=True`, which can be useful for debugging - otherwise operator which
        does not contribute to the pipeline output is removed from the graph.
        """

        prev = Pipeline.current()
        pipeline_tls.current_pipeline = pipeline
        stack = getattr(pipeline_tls, 'pipeline_stack', None)
        if stack is None:
            pipeline_tls.pipeline_stack = [prev]
        else:
            stack.append(prev)
        return prev

    @staticmethod
    def pop_current():
        """Restores previous pipeline as current. Complementary to :meth:`push_current`."""
        pipeline_tls.current_pipeline = pipeline_tls.pipeline_stack.pop()

    def __enter__(self):
        """Safely sets the pipeline as current.
        Current pipeline is required to call operators with side effects or without outputs.
        Examples of such operators are `PythonFunction` (potential side effects) or `DumpImage`
        (no output).

        Any dangling operator can be marked as having side effects if it's marked
        with `preserve=True`, which can be useful for debugging - otherwise operator which
        does not contribute to the pipeline output is removed from the graph.

        To manually set new (and restore previous) current pipeline, use :meth:`push_current`
        and :meth:`pop_current`, respectively.
        """
        Pipeline.push_current(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Safely restores previous pipeline."""
        Pipeline.pop_current()


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
        if self._api_type is None:
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
    def _prepare_graph(self, define_graph = None):
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
        self._pipe.EnableExecutorMemoryStats(self._enable_memory_stats)

        if define_graph is not None:
            if self._graph_out is not None:
                raise RuntimeError("Duplicate graph definition - `define_graph` argument "
                    "should not be specified when graph was defined with a call to `set_outputs`.")
        else:
            define_graph = self.define_graph

        if self._graph_out:
            outputs = self._graph_out
        else:
            with self:
                outputs = define_graph()
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        elif not isinstance(outputs, list):
            outputs = [outputs]

        for i in range(len(outputs)):
            if isinstance(outputs[i], types.ScalarConstant):
                import nvidia.dali.ops
                outputs[i] = nvidia.dali.ops._instantiate_constant_node("cpu", outputs[i])
            _data_node._check(outputs[i])

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
        self._ops = []
        while ops:
            op = ops.pop()
            self._ops.append(op)
            if op.relation_id not in related_logical_id:
                related_logical_id[op.relation_id] = self._pipe.AddOperator(op.spec, op.name)
            else:
                self._pipe.AddOperator(op.spec, op.name, related_logical_id[op.relation_id])
        self._prepared = True
        self._setup_input_callbacks()
        self._names_and_devices = [(e.name, e.device) for e in outputs]

    def _setup_input_callbacks(self):
        from nvidia.dali.external_source import _is_external_source_with_callback
        groups = set()
        for op in self._ops:
            if _is_external_source_with_callback(op):
                group = op._group
                groups.add(group)
        self._input_callbacks = list(groups)

    def build(self, define_graph = None):
        """Build the pipeline.

        Pipeline needs to be built in order to run it standalone.
        Framework-specific plugins handle this step automatically.

        Parameters
        ----------
        define_graph : callable
            If specified, this function will be used instead of member :meth:`define_graph`.
            This parameter must not be set, if the pipeline outputs are specified with
            :meth:`set_outputs`.
        """
        if self._built:
            return

        if self.num_threads < 1:
            raise ValueError("Pipeline created with `num_threads` < 1 can only be used "
                             "for serialization.")

        if not self._prepared:
            self._prepare_graph(define_graph)

        self._pipe.Build(self._names_and_devices)
        self._built = True

    def feed_input(self, data_node, data, layout="", cuda_stream = None):
        """Pass a mutlidimensional array or DLPack (or a list thereof) to an output of ExternalSource.
        In the case of the GPU input, the data must be modified on the same stream as the one
        used by feed_input. See ``cuda_stream`` parameter for details.

        Parameters
        ----------
        data_node : :class:`DataNode` or str
            The name of the :class:`nvidia.dali.ops.ExternalSource` node or a :class:`DataNode`
            object returned by a call to that ExternalSource.

        data : an ndarray or DLPack or a list thereof
            The array(s) may be one of:
              * NumPy ndarray (CPU)
              * MXNet ndarray (CPU)
              * PyTorch tensor (CPU or GPU)
              * CuPy array (GPU)
              * objects implementing ``__cuda_array_interface__``
            The data to be used as the output of the ExternalSource referred to by `data_node`.

        layout : str
            The description of the data layout (or empty string, if not specified).
            It should be a string of the length that matches the dimensionality of the data, batch
            dimension excluded. For a batch of channel-first images, this should be "CHW", for
            channel-last video it's "FHWC" and so on.

        cuda_stream : optional, `cudaStream_t` or an object convertible to `cudaStream_t`, e.g. `cupy.cuda.Stream`, `torch.cuda.Stream`
            The CUDA stream, which is going to be used for copying data to GPU or from a GPU
            source. If not set, best effort will be taken to maintain correctness - i.e. if the data
            is provided as a tensor/array from a recognized library (CuPy, PyTorch), the library's
            current stream is used. This should work in typical scenarios, but advanced use cases
            (and code using unsupported libraries) may still need to supply the stream handle
            explicitly.

            Special values:
              *  0 - use default CUDA stream
              * -1 - use DALI's internal stream

            If internal stream is used, the call to ``feed_input`` will block until the copy to
            internal buffer is complete, since there's no way to synchronize with this stream to
            prevent overwriting the array with new data in another stream.
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if isinstance(data_node, str):
            name = data_node
        else:
            _data_node._check(data_node)
            name = data_node.name

        from nvidia.dali.external_source import _check_data_batch

        infer_stream = False
        if cuda_stream is None:
            infer_stream = True
        if cuda_stream == -1:
            cuda_stream = None
        else:
            cuda_stream = types._raw_cuda_stream(cuda_stream)

        def to_numpy(x):
            if types._is_mxnet_array(x):
                return x.asnumpy()
            elif types._is_torch_tensor(x):
                return x.numpy()
            else:
                return x

        # __cuda_array_interface__ doesn't provide any way to pass the information about the device
        # where the memory is located. It is assumed that the current device is the one that the memory belongs to,
        # unless the user sets the device explicitly creating TensorGPU/TensorListGPU
        if isinstance(data, list):
            inputs = []
            checked = False
            for datum in data:
                info = CheckDLPackCapsule(datum)
                if not info[0] and not checked:
                    _check_data_batch(data, self._batch_size, layout)
                    checked = True
                if hasattr(datum, "__cuda_array_interface__") or (info[0] and info[1]):
                    if infer_stream:
                        cuda_stream = _get_default_stream_for_array(datum)
                    inp = Tensors.TensorGPU(datum, layout)
                else:
                    datum = to_numpy(datum)
                    inp = Tensors.TensorCPU(datum, layout)
                inputs.append(inp)
            assert all(isinstance(inp, type(inputs[0])) for inp in inputs), \
                   "Mixed input types are not support, all need to reside on the CPU or GPU"
            self._pipe.SetExternalTensorInput(name, inputs, ctypes.c_void_p(cuda_stream))
        else:
            info = CheckDLPackCapsule(data)
            if not info[0]:
                _check_data_batch(data, self._batch_size, layout)
            if hasattr(data, "__cuda_array_interface__") or (info[0] and info[1]):
                if infer_stream:
                    cuda_stream = _get_default_stream_for_array(data)
                inp = Tensors.TensorListGPU(data, layout)
            else:
                data = to_numpy(data)
                inp = Tensors.TensorListCPU(data, layout)
            self._pipe.SetExternalTLInput(name, inp, ctypes.c_void_p(cuda_stream))

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
        reached its end - usually when iter_setup cannot produce any more data.

        :return:
            A list of `TensorList` objects for respective pipeline outputs
        """
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
        Needs to be used together with :meth:`release_outputs`
        and :meth:`share_outputs`.
        Should not be mixed with :meth:`run` in the same pipeline"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._first_iter and self._exec_pipelined:
                self._prefetch()
            else:
                self._run_once()

    # for the backward compatibility
    def _run(self):
        """Deprecated. Use `schedule_run` instead."""
        _show_deprecation_warning("_run", "schedule_run")
        self.schedule_run()

    def share_outputs(self):
        """Returns the outputs of the pipeline.

        Main difference to :meth:`outputs`
        is that share_outputs doesn't release returned buffers, release_outputs
        need to be called for that. If the pipeline is executed asynchronously,
        this function blocks until the results become available. It provides
        the user with better control about when he wants to run the pipeline, when he wants
        to obtain the resulting buffers and when they can be returned to DALI pool when the
        results have been consumed.
        Needs to be used together with :meth:`release_outputs`
        and :meth:`schedule_run`
        Should not be mixed with :meth:`run` in the same pipeline.

        :return:
            A list of `TensorList` objects for respective pipeline outputs
        """
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._batches_to_consume == 0 or self._gpu_batches_to_consume == 0:
                raise StopIteration
            self._batches_to_consume -= 1
            self._gpu_batches_to_consume -= 1
            return self._pipe.ShareOutputs()

    # for the backward compatibility
    def _share_outputs(self):
        """Deprecated. Use :meth:`share_outputs` instead"""
        _show_deprecation_warning("_share_outputs", "share_outputs")
        self.share_outputs()

    def release_outputs(self):
        """Release buffers returned by share_outputs calls.

        It helps in case when output call result is consumed (copied)
        and buffers can be marked as free before the next call to share_outputs. It provides
        the user with better control about when he wants to run the pipeline, when he wants
        to obtain the resulting buffers and when they can be returned to DALI pool when the
        results have been consumed.
        Needs to be used together with :meth:`schedule_run`
        and :meth:`share_outputs`
        Should not be mixed with :meth:`run` in the same pipeline"""
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if not self._built:
                raise RuntimeError("Pipeline must be built first.")
            return self._pipe.ReleaseOutputs()

    # for the backward compatibility
    def _release_outputs(self):
        """Deprecated. Use :meth:`release_outputs` instead"""
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
        Should not be mixed with :meth:`schedule_run` in the same pipeline,
        :meth:`share_outputs` and
        :meth:`release_outputs`

        :return:
            A list of `TensorList` objects for respective pipeline outputs
        """
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
                self._iter_setup()
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
                self._iter_setup()
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
            self._iter = 0

    def empty(self):
        """If there is any work scheduled in the pipeline but not yet consumed
        """
        return self._batches_to_consume == 0

    def serialize(self, define_graph=None, filename=None):
        """Serialize the pipeline to a Protobuf string.

        Additionally, you can pass file name, so that serialized pipeline will be written there.
        The file contents will be overwritten

        Parameters
        ----------
        define_graph : allable
                If specified, this function will be used instead of member :meth:`define_graph`.
                This parameter must not be set, if the pipeline outputs are specified with
                :meth:`set_outputs`.
        filename : str
                File, from where serialized pipeline will be writeen.
        kwargs : dict
                Refer to Pipeline constructor for full list of arguments.
        """
        if not self._prepared:
            self._prepare_graph(define_graph)
            self._pipe.SetOutputNames(self._names_and_devices)
        ret = self._pipe.SerializeToProtobuf()
        if filename is not None:
            with open(filename, 'wb') as pipeline_file:
                pipeline_file.write(ret)
        return ret

    @classmethod
    def deserialize(cls, serialized_pipeline=None, filename=None, **kwargs):
        """Deserialize and build pipeline.

        Deserialize pipeline, previously serialized with ``serialize()`` method.

        Returned pipeline is already built.

        Alternatively, additional arguments can be passed, which will be used when instantiating
        the pipeline. Refer to Pipeline constructor for full list of arguments. By default,
        the pipeline will be instantiated with the arguments from serialized pipeline.

        Note, that ``serialized_pipeline`` and ``filename`` parameters are mutually exclusive

        Parameters
        ----------
        serialized_pipeline : str
                   Pipeline, serialized using ``serialize()`` method.
        filename : str
                   File, from which serialized pipeline will be read.
        kwargs : dict
                   Refer to Pipeline constructor for full list of arguments.

        Returns
        ----------
        Deserialized and built pipeline.
        """
        kw = kwargs
        if (serialized_pipeline is None) == (filename is None):  # XNOR
            raise ValueError(
                "serialized_pipeline and filename arguments are mutually exclusive. "
                "Precisely one of them should be defined.")
        pipeline = cls()
        if filename is not None:
            with open(filename, 'rb') as pipeline_file:
                serialized_pipeline = pipeline_file.read()
        pipeline._pipe = b.Pipeline(
            serialized_pipeline,
            kw.get("batch_size", -1),
            kw.get("num_threads", -1),
            kw.get("device_id", -1),
            kw.get("exec_pipelined", True),
            kw.get("prefetch_queue_depth", 2),
            kw.get("exec_async", True),
            kw.get("bytes_per_sample", 0),
            kw.get("set_affinity", False),
            kw.get("max_streams", -1),
            kw.get("default_cuda_stream_priority", 0)
        )
        pipeline._pipe.SetExecutionTypes(pipeline._exec_pipelined, pipeline._exec_separated,
                                         pipeline._exec_async)
        pipeline._pipe.SetQueueSizes(pipeline._cpu_queue_size, pipeline._gpu_queue_size)
        pipeline._pipe.EnableExecutorMemoryStats(pipeline._enable_memory_stats)
        pipeline._prepared = True
        pipeline._pipe.Build()
        pipeline._built = True
        return pipeline

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
        self._pipe.EnableExecutorMemoryStats(self._enable_memory_stats)
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

    def set_outputs(self, *output_data_nodes):
        """Set the outputs of the pipeline.

        Use of this function is an alternative to overriding `define_graph` in a derived class.

        Args
        ----
        `*output_data_nodes` : unpacked list of :class:`DataNode` objects
            The outputs of the pipeline
        """
        self._graph_out = output_data_nodes

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.

        It returns a list of outputs created by calling DALI Operators."""
        raise NotImplementedError

    def _run_input_callbacks(self):
        if self._input_callbacks is None:
            return

        for group in self._input_callbacks:
            group.call_and_feed(self, self._iter)

    def _iter_setup(self):
        self._run_input_callbacks()
        self.iter_setup()
        self._iter += 1

    def iter_setup(self):
        """This function can be overriden by user-defined
        pipeline to perform any needed setup for each iteration.
        For example, one can use this function to feed the input
        data from NumPy arrays."""
        pass
