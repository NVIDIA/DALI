# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=no-member
from collections import deque
from nvidia.dali import backend as b
from nvidia.dali import tensors as Tensors
from nvidia.dali import types
from nvidia.dali._multiproc.pool import WorkerPool
from nvidia.dali import pickling as dali_pickle
from nvidia.dali.backend import CheckDLPackCapsule
from threading import local as tls
from . import data_node as _data_node
import functools
import inspect
import warnings
import weakref
import ctypes

pipeline_tls = tls()

from .data_node import DataNode

DataNode.__module__ = __name__  # move to pipeline


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
    Maximum batch size of the pipeline. Negative values for this parameter
    are invalid - the default value may only be used with
    serialized pipeline (the value stored in serialized pipeline
    is used instead). In most cases, the actual batch size of the pipeline
    will be equal to the maximum one. Running the DALI Pipeline with a smaller batch size
    is also supported. The batch size might change from iteration to iteration.

    Please note, that DALI might perform memory preallocations according to this
    parameter. Setting it too high might result in out-of-memory failure.
`num_threads` : int, optional, default = -1
    Number of CPU threads used by the pipeline.
    Negative values for this parameter are invalid - the default
    value may only be used with serialized pipeline (the value
    stored in serialized pipeline is used instead).
`device_id` : int, optional, default = -1
    Id of GPU used by the pipeline.
    A None value for this parameter means that DALI should not use GPU nor CUDA runtime.
    This limits the pipeline to only CPU operators but allows it to run on any CPU capable machine.
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
`enable_memory_stats`: bool, optional, default = 1
    If DALI should print operator output buffer statistics.
    Usefull for `bytes_per_sample_hint` operator parameter.
`py_num_workers`: int, optional, default = 1
    The number of Python workers that will process ``ExternalSource`` callbacks.
    The pool starts only if there is at least one ExternalSource with ``parallel`` set to True.
    Setting it to 0 disables the pool and all ExternalSource operators fall back to non-parallel
    mode even if ``parallel`` is set to True.
`py_start_method` : str, default = "fork"
    Determines how Python workers are started. Supported methods:

      * ``"fork"`` - start by forking the process
      * ``"spawn"`` - start a fresh interpreter process

    If ``spawn`` method is used, ExternalSource's callback must be picklable.
    In order to use ``fork``, there must be no CUDA contexts acquired at the moment of starting
    the workers. For this reason, if you need to build multiple pipelines that use Python workers,
    you will need to call :meth:`start_py_workers` before calling :meth:`build` of any
    of the pipelines. You can find more details and caveats of both methods in Python's
    ``multiprocessing`` module documentation.
`py_callback_pickler` : module or tuple, default = None
    If `py_start_method` is set to *spawn*, callback passed to parallel ExternalSource must be picklable.
    If run in Python3.8 or newer with `py_callback_pickler` set to None, DALI uses customized pickle
    when serializing callbacks to support serialization of local functions and lambdas.

    However, if you need to serialize more complex objects like local classes or you are running
    older version of Python you can provide external serialization package such as dill or cloudpickle
    that implements two methods: `dumps` and `loads` to make DALI use them to serialize
    external source callbacks. You can pass a module directly as ``py_callback_pickler``::
        
        import dill
        @pipeline_def(py_callback_pickler=dill, ...)
        def create_pipeline():
            src = fn.external_source(lambda sample_info: np.int32([42]), batch=False, parallel=True)
            ...

    A valid value for `py_callback_pickler` is either a module/object implementing
    ``dumps`` and ``loads`` methods or a tuple where the first item is the module/object and the next
    two optional parameters are extra kwargs to be passed when calling dumps and loads respectively.
    The provided methods and kwargs must be picklable with standard `pickle.dumps`.

    If you run Python3.8 or newer with the default DALI pickler (`py_callback_pickler` = None),
    you can hint DALI to serialize global functions by value rather than by reference
    by decorating them with `@dali.pickling.pickle_by_value`. It may be especially useful when
    working with Jupyter notebook to work around the issue of worker process being unable to import
    the callback defined as a global function inside the notebook.
"""
    def __init__(self, batch_size = -1, num_threads = -1, device_id = -1, seed = -1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 set_affinity=False, max_streams=-1, default_cuda_stream_priority = 0,
                 *,
                 enable_memory_stats=False, py_num_workers=1, py_start_method="fork",
                 py_callback_pickler=None):
        self._sinks = []
        self._max_batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        # When initializing DALI, we do the following in order:
        # * Discover the ops specified in Python, group the ExternalSources (_build_graph())
        # * Start the Python workers pool (_start_py_workers())
        # * Construct the C++ Pipeline backend and pass the graph to it (_init_pipeline_backend())
        # * Build the pieline. (_pipe.Build())
        # In case of deserialized pipeline, only _backend_prepared and _built will be True
        self._py_graph_built = False
        self._py_pool_started = False
        self._backend_prepared = False
        self._built = False
        self._first_iter = True
        self._last_iter = False
        self._iter = 0
        self._batches_to_consume = 0
        self._cpu_batches_to_consume = 0
        self._gpu_batches_to_consume = 0
        self._names_and_devices = None
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._set_affinity = set_affinity
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._py_num_workers = py_num_workers
        self._py_start_method = py_start_method
        if py_callback_pickler is not None and py_start_method == "fork":
            raise ValueError("``py_callback_pickler`` should not be set when 'fork' start method is used.")
        if py_callback_pickler is None and py_start_method == "spawn":
           py_callback_pickler = dali_pickle._DaliPickle
        self._py_callback_pickler = py_callback_pickler
        self._api_type = None
        self._skip_api_check = False
        self._graph_out = None
        self._ops = None
        self._graph_outputs = None
        self._py_pool = None
        self._input_callbacks = None
        self._parallel_input_callbacks = None
        self._seq_input_callbacks = None
        self._enable_memory_stats = enable_memory_stats
        self._prefetch_queue_depth = prefetch_queue_depth
        if type(prefetch_queue_depth) is dict:
            self._exec_separated = True
            self._cpu_queue_size = prefetch_queue_depth["cpu_size"]
            self._gpu_queue_size = prefetch_queue_depth["gpu_size"]
        elif type(prefetch_queue_depth) is int:
            self._exec_separated = False
            self._cpu_queue_size = prefetch_queue_depth
            self._gpu_queue_size = prefetch_queue_depth
        else:
            raise TypeError("Expected prefetch_queue_depth to be either int or Dict[int, int]")

    @property
    def batch_size(self):
        """Batch size."""
        _show_deprecation_warning("batch_size", "max_batch_size")
        return self._max_batch_size

    @property
    def max_batch_size(self):
        """Maximum batch size."""
        return self._max_batch_size

    @property
    def num_threads(self):
        """Number of CPU threads used by this pipeline."""
        return self._num_threads

    @property
    def device_id(self):
        """Id of the GPU used by the pipeline or None for CPU-only pipelines."""
        return None if self._device_id == types.CPU_ONLY_DEVICE_ID else self._device_id

    @property
    def seed(self):
        """Random seed used in the pipeline or None, if seed is not fixed."""
        return self._seed

    @property
    def exec_pipelined(self):
        """If true, pipeline execution model is used."""
        return self._exec_pipelined

    @property
    def exec_async(self):
        """If true, asynchronous execution is used."""
        return self._exec_async

    @property
    def set_affinity(self):
        """If True, worker threads are bound to CPU cores."""
        return self._set_affinity

    @property
    def max_streams(self):
        """Reserved for future use."""
        return self._max_streams

    @property
    def prefetch_queue_depth(self):
        """Depth (or depths) of the prefetch queue, as specified in the ``__init__`` arguments."""
        return self._prefetch_queue_depth

    @property
    def default_cuda_stream_priority(self):
        """Default priority of the CUDA streams used by this pipeline."""
        return self._default_cuda_stream_priority

    @property
    def enable_memory_stats(self):
        """If True, memory usage statistics are gathered."""
        return self._enable_memory_stats

    @property
    def py_num_workers(self):
        """The number of Python worker processes used by parallel ```external_source```."""
        return self._py_num_workers

    @property
    def py_start_method(self):
        """The method of launching Python worker processes used by parallel ```external_source```."""
        return self._py_start_method

    @property
    def exec_separated(self):
        """If True, there are separate prefetch queues for CPU and GPU stages."""
        return self._exec_separated

    @property
    def cpu_queue_size(self):
        """The number of iterations processed ahead by the CPU stage."""
        return self._cpu_queue_size

    @property
    def gpu_queue_size(self):
        """The number of iterations processed ahead by the GPU stage."""
        return self._gpu_queue_size

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
        meta = self._pipe.reader_meta()
        return {k : v["epoch_size_padded"] for k, v in meta.items()}

    def executor_statistics(self):
        """Returns provided pipeline executor statistics metadata as a dictionary.
        Each key in the dictionary is the operator name. To enable it use ``executor_statistics``

        Available metadata keys for each operator:

            * ``real_memory_size`` - list of memory sizes that is used by each output of the operator.
              Index in the list corresponds to the output index.

            * ``max_real_memory_size`` - list of maximum tensor size that is used by each output of the operator.
              Index in the list corresponds to the output index.

            * ``reserved_memory_size`` - list of memory sizes that is reserved for each of the operator outputs.
              Index in the list corresponds to the output index.

            * ``max_reserved_memory_size`` - list of maximum memory sizes per tensor that is reserved for each of the operator outputs.
              Index in the list corresponds to the output index.
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
        """Returns the instance of the current pipeline set by :meth:`push_current`."""
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
    def _build_graph(self, define_graph=None):
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

        def is_nested(container, pred, container_types):
            if isinstance(container, container_types):
                if any(pred(x) for x in container):
                    return True
                for x in container:
                    if is_nested(x, pred, container_types):
                        return True
            return False

        def contains_nested_datanode(nested):
            return is_nested(nested, lambda x: isinstance(x, DataNode), (list, tuple))

        for i in range(len(outputs)):
            if isinstance(outputs[i], types.ScalarConstant):
                import nvidia.dali.ops
                outputs[i] = nvidia.dali.ops._instantiate_constant_node("cpu", outputs[i])
            elif contains_nested_datanode(outputs[i]):
                raise TypeError(f"Illegal pipeline output type. The output {i} contains a nested "
                                "`DataNode`. Missing list/tuple expansion (*) is the likely cause.")
            elif not isinstance(outputs[i], DataNode):
                try:
                    outputs[i] = types.Constant(outputs[i], device="cpu")
                except TypeError:
                    raise TypeError(f"Illegal output type. The output {i} is a `{type(outputs[i])}`. "
                                     "Allowed types are ``DataNode`` and types convertible to "
                                     "`types.Constant` (numerical constants, 1D lists/tuple of numbers "
                                     "and ND arrays).")

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
        ops.reverse()
        self._ops = ops
        self._graph_outputs = outputs
        self._setup_input_callbacks()
        self._py_graph_built = True

    def _setup_pipe_pool_dependency(self):
        if self._py_pool_started:
            # The sole point of this call is to ensure the lifetime of the pool exceeds the lifetime
            # of the pipeline's backend, so that shared memory managed by the pool is not freed
            # before pipline's backend is garbage collected.
            # Otherwise the backend may try to access unmmaped memory which leads to crashes at the Python teardown.
            self._pipe.SetPyObjDependency(self._py_pool)

    def _start_py_workers(self):
        if not self._parallel_input_callbacks:
            return
        self._py_pool = WorkerPool.from_groups(
            self._parallel_input_callbacks, self._prefetch_queue_depth, self._py_start_method,
            self._py_num_workers, py_callback_pickler=self._py_callback_pickler)
        # ensure processes started by the pool are termineted when pipeline is no longer used
        weakref.finalize(self, lambda pool : pool.close(), self._py_pool)
        self._py_pool_started = True

    def _init_pipeline_backend(self):
        self._pipe = b.Pipeline(self._max_batch_size,
                                self._num_threads,
                                self._device_id if self._device_id is not None else types.CPU_ONLY_DEVICE_ID,
                                self._seed if self._seed is not None else -1,
                                self._exec_pipelined,
                                self._cpu_queue_size,
                                self._exec_async,
                                self._bytes_per_sample,
                                self._set_affinity,
                                self._max_streams,
                                self._default_cuda_stream_priority)
        self._pipe.SetExecutionTypes(self._exec_pipelined, self._exec_separated, self._exec_async)
        self._pipe.SetQueueSizes(self._cpu_queue_size, self._gpu_queue_size)
        self._pipe.EnableExecutorMemoryStats(self._enable_memory_stats)

        # Add the ops to the graph and build the backend
        related_logical_id = {}

        for op in self._ops:
            if op.relation_id not in related_logical_id:
                related_logical_id[op.relation_id] = self._pipe.AddOperator(op.spec, op.name)
            else:
                self._pipe.AddOperator(op.spec, op.name, related_logical_id[op.relation_id])
        self._backend_prepared = True
        self._names_and_devices = [(e.name, e.device) for e in self._graph_outputs]

    def _setup_input_callbacks(self):
        from nvidia.dali.external_source import _is_external_source_with_callback
        groups = set()
        for op in self._ops:
            if _is_external_source_with_callback(op):
                group = op._group
                groups.add(group)
        groups = list(groups)
        self._input_callbacks = groups
        if self._py_num_workers == 0:
            self._parallel_input_callbacks = []
            self._seq_input_callbacks = self._input_callbacks
        else:
            self._parallel_input_callbacks = [group for group in groups if group.parallel]
            self._seq_input_callbacks = [group for group in groups if not group.parallel]

    def start_py_workers(self):
        """
        Start Python workers (that will run ``ExternalSource`` callbacks).
        You need to call :meth:`start_py_workers` before you call any functionality that creates
        or acquires CUDA context when using ``fork`` to start Python workers (``py_start_method="fork"``).
        It is called automatically by :meth:`Pipeline.build` method when such separation is not necessary.

        If you are going to build more than one pipeline that starts Python workers by forking
        the process then you need to call :meth:`start_py_workers` method on all those pipelines
        before calling :meth:`build` method of any pipeline, as build acquires CUDA context
        for current process.

        The same applies to using any other functionality that would create CUDA context -
        for example, initializing a framework that uses CUDA or creating CUDA tensors with it.
        You need to call :meth:`start_py_workers` before you call such functionality when
        using ``py_start_method="fork"``.

        Forking a process that has a CUDA context is unsupported and may lead to unexpected errors.

        If you use the method you cannot specify ``define_graph`` argument when calling :meth:`build`.
        """
        if not self._py_graph_built:
            self._build_graph()
        if not self._py_pool_started:
            self._start_py_workers()

    def build(self, define_graph = None):
        """Build the pipeline.

        Pipeline needs to be built in order to run it standalone.
        Framework-specific plugins handle this step automatically.

        Parameters
        ----------
        define_graph : callable
            If specified, this function will be used instead of member :meth:`define_graph`.
            This parameter must not be set, if the pipeline outputs are specified with
            :meth:`set_outputs` or if the :meth:`start_py_workers` is used.

            .. note::

                This method of defining the processing graph cannot be used with parallel
                ``ExternalSource``.
        """
        if self._built:
            return

        if self.num_threads < 1:
            raise ValueError("Pipeline created with `num_threads` < 1 can only be used "
                             "for serialization.")

        self.start_py_workers()
        if not self._backend_prepared:
            self._init_pipeline_backend()
        self._setup_pipe_pool_dependency()

        self._pipe.Build(self._names_and_devices)
        self._built = True

    def feed_input(self, data_node, data, layout = None, cuda_stream = None, use_copy_kernel = False):
        """Pass a mutlidimensional array or DLPack (or a list thereof) to an output of ExternalSource.
        In the case of the GPU input, the data must be modified on the same stream as the one
        used by feed_input. See ``cuda_stream`` parameter for details.

        Parameters
        ----------
        data_node : :class:`DataNode` or str
            The name of the :class:`nvidia.dali.fn.external_source` node or a :class:`DataNode`
            object returned by a call to that ExternalSource.

        data : an ndarray or DLPack or a list thereof
            The array(s) may be one of:

              * NumPy ndarray (CPU)
              * MXNet ndarray (CPU)
              * PyTorch tensor (CPU or GPU)
              * CuPy array (GPU)
              * objects implementing ``__cuda_array_interface__``
              * DALI `TensorList` or list of DALI `Tensor` objects

            The data to be used as the output of the ExternalSource referred to by `data_node`.

        layout : str or None
            The description of the data layout (or empty string, if not specified).
            It should be a string of the length that matches the dimensionality of the data, batch
            dimension excluded. For a batch of channel-first images, this should be "CHW", for
            channel-last video it's "FHWC" and so on.
            If ``data`` is a DALI `TensorList` or a list of DALI `Tensor` objects and ``layout``
            is ``None``, the layout is taken from ``data``.

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

        use_copy_kernel : optional, `bool`
            If set to True, DALI will use a CUDA kernel to feed the data (only applicable when copying
            data to/from GPU memory) instead of cudaMemcpyAsync (default).
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
        if isinstance(data, (Tensors.TensorListCPU, Tensors.TensorListGPU)):
            if layout is not None:
                _check_data_batch(data, self._max_batch_size, layout)
                data = type(data)(data, layout)

            self._pipe.SetExternalTLInput(name, data, ctypes.c_void_p(cuda_stream), use_copy_kernel)
        elif isinstance(data, list):
            inputs = []
            checked = False
            for datum in data:
                info = CheckDLPackCapsule(datum)
                if not info[0] and not checked:
                    _check_data_batch(data, self._max_batch_size, layout)
                    checked = True
                if isinstance(datum, (Tensors.TensorCPU, Tensors.TensorGPU)):
                    inp = type(datum)(datum, layout=layout) if layout is not None else datum
                elif hasattr(datum, "__cuda_array_interface__") or (info[0] and info[1]):
                    if infer_stream:
                        cuda_stream = _get_default_stream_for_array(datum)
                    inp = Tensors.TensorGPU(datum, layout or "")
                else:
                    datum = to_numpy(datum)
                    inp = Tensors.TensorCPU(datum, layout or "")
                inputs.append(inp)
            assert all(isinstance(inp, type(inputs[0])) for inp in inputs), \
                   "Mixed input types are not support, all need to reside on the CPU or GPU"
            self._pipe.SetExternalTensorInput(name, inputs, ctypes.c_void_p(cuda_stream), use_copy_kernel)
        else:
            info = CheckDLPackCapsule(data)
            if not info[0]:
                _check_data_batch(data, self._max_batch_size, layout)
            if hasattr(data, "__cuda_array_interface__") or (info[0] and info[1]):
                if infer_stream:
                    cuda_stream = _get_default_stream_for_array(data)
                inp = Tensors.TensorListGPU(data, layout or "")
            else:
                data = to_numpy(data)
                inp = Tensors.TensorListCPU(data, layout or "")
            self._pipe.SetExternalTLInput(name, inp, ctypes.c_void_p(cuda_stream), use_copy_kernel)

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
        self._schedule_py_workers()
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

    def _schedule_py_workers(self):
        if self._py_pool is None:
            return
        for i, group in enumerate(self._parallel_input_callbacks):
            group.prefetch(self._py_pool, i, self._max_batch_size)

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
            if self._input_callbacks:
                for group in self._input_callbacks:
                    group.reset_indices()
            if self._py_pool:
                self._py_pool.reset()

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
        define_graph : callable
                If specified, this function will be used instead of member :meth:`define_graph`.
                This parameter must not be set, if the pipeline outputs are specified with
                :meth:`set_outputs`.
        filename : str
                File, from where serialized pipeline will be writeen.
        kwargs : dict
                Refer to Pipeline constructor for full list of arguments.
        """
        if define_graph is not None and not callable(define_graph):
            raise TypeError(
                "Provided `define_graph` argument is not callable." +
                (" Didn't you want to write `.serialize(filename=...)`?"
                if isinstance(define_graph, str) else ""))
        if not self._py_graph_built:
            self._build_graph(define_graph)
        if not self._backend_prepared:
            self._init_pipeline_backend()
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
        pipeline._backend_prepared = True
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
                                self._max_batch_size,
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
        self._backend_prepared = True
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

        batches = []   # data from external source callbacks is gathered here
        stop_iter = False
        for i, group in enumerate(self._parallel_input_callbacks):
            try:
                batches.append(group.schedule_and_receive(self, self._py_pool, i, self._max_batch_size))
            except StopIteration:
                stop_iter = True
        for group in self._seq_input_callbacks:
            try:
                batches.append(group.get_batch(self, self._max_batch_size))
            except StopIteration:
                stop_iter = True
        if stop_iter:
            raise StopIteration()

        # we only fill external source queues when we know that all callbacks succeeded
        for batch in batches:
            batch.feed()

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


def _discriminate_args(func, **func_kwargs):
    """Split args on those applicable to Pipeline constructor and the decorated function."""
    func_argspec = inspect.getfullargspec(func)
    ctor_argspec = inspect.getfullargspec(Pipeline.__init__)

    ctor_args = {}
    fn_args = {}

    if func_argspec.varkw is not None:
        raise TypeError(
            "Using variadic keyword argument `**{}` in graph-defining function is not allowed.".format(
                func_argspec.varkw))

    for farg in func_kwargs.items():
        is_ctor_arg = farg[0] in ctor_argspec.args or farg[0] in ctor_argspec.kwonlyargs
        is_fn_arg = farg[0] in func_argspec.args or farg[0] in func_argspec.kwonlyargs
        if is_fn_arg:
            fn_args[farg[0]] = farg[1]
            if is_ctor_arg:
                print(
                    "Warning: the argument `{}` shadows a Pipeline constructor argument of the same name.".format(
                        farg[0]))
        elif is_ctor_arg:
            ctor_args[farg[0]] = farg[1]
        else:
            assert False, "This shouldn't happen. Please double-check the `{}` argument".format(farg[0])

    return ctor_args, fn_args


def pipeline_def(fn=None, **pipeline_kwargs):
    """
    Decorator that converts a graph definition function into a DALI pipeline factory.

    A graph definition function is a function that returns intended pipeline outputs.
    You can decorate this function with ``@pipeline_def``::

        @pipeline_def
        def my_pipe(flip_vertical, flip_horizontal):
            ''' Creates a DALI pipeline, which returns flipped and original images '''
            data, _ = fn.readers.file(file_root=images_dir)
            img = fn.decoders.image(data, device="mixed")
            flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
            return flipped, img

    The decorated function returns a DALI Pipeline object::

        pipe = my_pipe(True, False)
        # pipe.build()  # the pipeline is not configured properly yet

    A pipeline requires additional parameters such as batch size, number of worker threads,
    GPU device id and so on (see :meth:`nvidia.dali.Pipeline()` for a complete list of pipeline parameters).
    These parameters can be supplied as additional keyword arguments, passed to the decorated function::

        pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)
        pipe.build()  # the pipeline is properly configured, we can build it now

    The outputs from the original function became the outputs of the Pipeline::

        flipped, img = pipe.run()

    When some of the pipeline parameters are fixed, they can be specified by name in the decorator::

        @pipeline_def(batch_size=42, num_threads=3)
        def my_pipe(flip_vertical, flip_horizontal):
            ...

    Any Pipeline constructor parameter passed later when calling the decorated function will
    override the decorator-defined params::

        @pipeline_def(batch_size=32, num_threads=3)
        def my_pipe():
            data = fn.external_source(source=my_generator)
            return data

        pipe = my_pipe(batch_size=128)  # batch_size=128 overrides batch_size=32

    .. warning::

        The arguments of the function being decorated can shadow pipeline constructor arguments -
        in which case there's no way to alter their values.

    .. note::

        Using ``**kwargs`` (variadic keyword arguments) in graph-defining function is not allowed.
        They may result in unwanted, silent hijacking of some arguments of the same name by
        Pipeline constructor. Code written this way would cease to work with future versions of DALI
        when new parameters are added to the Pipeline constructor.

    To access any pipeline arguments within the body of a ``@pipeline_def`` function, the function
    :meth:`nvidia.dali.Pipeline.current()` can be used::

        @pipeline_def()
        def my_pipe():
            pipe = Pipeline.current()
            batch_size = pipe.batch_size
            num_threads = pipe.num_threads
            ...

        pipe = my_pipe(batch_size=42, num_threads=3)
        ...
    """
    def actual_decorator(func):
        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            ctor_args, fn_kwargs = _discriminate_args(func, **kwargs)
            pipe = Pipeline(**{**pipeline_kwargs, **ctor_args})  # Merge and overwrite dict
            with pipe:
                pipe_outputs = func(*args, **fn_kwargs)
                if isinstance(pipe_outputs, tuple):
                    po = pipe_outputs
                elif pipe_outputs is None:
                    po = ()
                else:
                    po = (pipe_outputs,)
                pipe.set_outputs(*po)
            return pipe
        return create_pipeline
    return actual_decorator(fn) if fn else actual_decorator
