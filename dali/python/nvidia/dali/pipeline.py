# Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List
from collections import deque
from nvidia.dali import backend as b
from nvidia.dali import types
from nvidia.dali import internal
from nvidia.dali._multiproc.pool import WorkerPool
from nvidia.dali import pickling as dali_pickle
from nvidia.dali import _conditionals
from threading import local as tls
from . import data_node as _data_node
import functools
import inspect
import warnings
import weakref
import ctypes
import sys
from .data_node import DataNode

pipeline_tls = tls()

DataNode.__module__ = __name__  # move to pipeline


def _show_deprecation_warning(deprecated, in_favor_of):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn("{} is deprecated, please use {} instead".format(deprecated, in_favor_of),
                      Warning, stacklevel=2)


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
    If `py_start_method` is set to *spawn*, callback passed to parallel
    ExternalSource must be picklable.
    If run in Python3.8 or newer with `py_callback_pickler` set to None, DALI uses customized pickle
    when serializing callbacks to support serialization of local functions and lambdas.

    However, if you need to serialize more complex objects like local classes or you are running
    older version of Python you can provide external serialization package such as dill or
    cloudpickle that implements two methods: `dumps` and `loads` to make DALI use them to serialize
    external source callbacks. You can pass a module directly as ``py_callback_pickler``::

        import dill
        @pipeline_def(py_callback_pickler=dill, ...)
        def create_pipeline():
            src = fn.external_source(lambda sample_info: np.int32([42]), batch=False, parallel=True)
            ...

    A valid value for `py_callback_pickler` is either a module/object implementing
    ``dumps`` and ``loads`` methods or a tuple where the first item is the module/object and
    the next two optional parameters are extra kwargs to be passed when calling dumps and
    loads respectively.
    The provided methods and kwargs must be picklable with standard `pickle.dumps`.

    If you run Python3.8 or newer with the default DALI pickler (`py_callback_pickler` = None),
    you can hint DALI to serialize global functions by value rather than by reference
    by decorating them with `@dali.pickling.pickle_by_value`. It may be especially useful when
    working with Jupyter notebook to work around the issue of worker process being unable to import
    the callback defined as a global function inside the notebook.
`output_dtype` : ``nvidia.dali.types.DALIDataType`` or list of those, default = None
    With this argument, you may declare, what data type you expect in the given output. You shall
    pass a list of mod:`types.DALIDataType`, each element in the list corresponding to
    one output from the pipeline. Additionally, you can pass ``None`` as a wildcard. The outputs,
    after each iteration, will be validated against the types you passed to this argument. If any
    output does not match the provided type, RuntimeError will be raised.

    If the ``output_dtype`` value is a single value (not a list), it will be broadcast to the
    number of outputs from the pipeline.
`output_ndim` : int or list of ints, default = None
    With this argument, you may declare, how many dimensions you expect in the given output.
    You shall pass a list of integers, each element in the list corresponding to one output
    from the pipeline.
    Additionally, you can pass ``None`` as a wildcard. The outputs, after each iteration, will be
    validated against the numbers of dimensions you passed to this argument. If the dimensionality
    of any output does not match the provided ``ndim``, RuntimeError will be raised.

    If the ``output_ndim`` value is a single value (not a list), it will be broadcast to the
    number of outputs from the pipeline.
"""

    def __init__(self,
                 batch_size=-1,
                 num_threads=-1,
                 device_id=-1,
                 seed=-1,
                 exec_pipelined=True,
                 prefetch_queue_depth=2,
                 exec_async=True,
                 bytes_per_sample=0,
                 set_affinity=False,
                 max_streams=-1,
                 default_cuda_stream_priority=0,
                 *,
                 enable_memory_stats=False,
                 py_num_workers=1,
                 py_start_method="fork",
                 py_callback_pickler=None,
                 output_dtype=None,
                 output_ndim=None):
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
        self._deserialized = False  # Marked True when deserializing
        self._first_iter = True
        self._last_iter = False
        self._iter = 0
        self._epoch_idx = 0
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
            raise ValueError(
                "``py_callback_pickler`` should not be set when 'fork' start method is used.")
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
        self._conditionals_enabled = False
        self._condition_stack = None

        # Assign and validate output_dtype
        if isinstance(output_dtype, (list, tuple)):
            for dtype in output_dtype:
                if not isinstance(dtype, (types.DALIDataType, type(None))):
                    raise TypeError(
                        f"`output_dtype` must be either: a value from "
                        f"nvidia.dali.types.DALIDataType, a list of these or None. "
                        f"Found type {type(dtype)} in the list."
                    )
                if dtype == types.NO_TYPE:
                    raise ValueError(
                        f"`output_dtype` can't be a types.NO_TYPE. Found {dtype} in the list.")
        elif not isinstance(output_dtype, (types.DALIDataType, type(None))):
            raise TypeError(
                f"`output_dtype` must be either: a value from nvidia.dali.types.DALIDataType, a "
                f"list of these or None. Found type: {type(output_dtype)}."
            )
        elif output_dtype == types.NO_TYPE:
            raise ValueError(
                f"`output_dtype` can't be a types.NO_TYPE. Found value: {output_dtype}")
        self._output_dtype = output_dtype

        # Assign and validate output_ndim
        if isinstance(output_ndim, (list, tuple)):
            for ndim in output_ndim:
                if not isinstance(ndim, (int, type(None))):
                    raise TypeError(
                        f"`output_ndim` must be either: an int, a list of ints or None. "
                        f"Found type {type(ndim)} in the list."
                    )
                if ndim is not None and ndim < 0:
                    raise ValueError(
                        f"`output_ndim` must be non-negative. Found value {ndim} in the list.")
        elif not isinstance(output_ndim, (int, type(None))):
            raise TypeError(
                f"`output_ndim` must be either: an int, a list of ints or None. "
                f"Found type: {type(output_ndim)}."
            )
        elif output_ndim is not None and output_ndim < 0:
            raise ValueError(f"`output_ndim` must be non-negative. Found value: {output_ndim}.")
        self._output_ndim = output_ndim

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
        """The method of launching Python worker processes used by
        parallel ```external_source```."""
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

    def output_dtype(self) -> list:
        """Data types expected at the outputs."""
        return [elem if elem != types.NO_TYPE else None for elem in self._pipe.output_dtype()]

    def output_ndim(self) -> list:
        """Number of dimensions expected at the outputs."""
        return [elem if elem != -1 else None for elem in self._pipe.output_ndim()]

    def epoch_size(self, name=None):
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
        return {k: v["epoch_size_padded"] for k, v in meta.items()}

    def executor_statistics(self):
        """Returns provided pipeline executor statistics metadata as a dictionary.
        Each key in the dictionary is the operator name. To enable it use ``executor_statistics``

        Available metadata keys for each operator:

            * ``real_memory_size`` - list of memory sizes that is used by each output of
              the operator. Index in the list corresponds to the output index.

            * ``max_real_memory_size`` - list of maximum tensor size that is used by each output of
              the operator. Index in the list corresponds to the output index.

            * ``reserved_memory_size`` - list of memory sizes that is reserved for each of
              the operator outputs. Index in the list corresponds to the output index.

            * ``max_reserved_memory_size`` - list of maximum memory sizes per tensor that is
              reserved for each of the operator outputs. Index in the list corresponds to
              the output index.
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        return self._pipe.executor_statistics()

    def external_source_shm_statistics(self):
        """Returns parallel external source's statistics regarding shared memory consumption.
        The returned dictionary contains following keys:

            * ``capacities`` - a list of sizes (in bytes) of shared memory slots allocated to
              accommodate data produced by the parallel external source.

            * ``per_sample_capacities`` - a list of sizes (in bytes) of shared memory slots
              divided by the mini-batch size, i.e. the maximal number of samples stored in
              such a slot. This value corresponds to external source's ``bytes_per_sample_hint``
              parameter, i.e., if the hint is big enough and the external source does not need
              to reallocate the memory, the values should be equal.
        """
        if self._py_pool is None:
            capacities, per_sample_capacities = [], []
        else:
            capacities = [
                shm.capacity for context in self._py_pool.contexts
                for shm in context.shm_manager.shm_pool
            ]
            per_sample_capacities = []
            for context in self._py_pool.contexts:
                num_mini_batches = context.shm_manager.num_minibatches
                batch_size = self.max_batch_size
                mini_batch_size = (batch_size + num_mini_batches - 1) // num_mini_batches
                for shm in context.shm_manager.shm_pool:
                    per_sample_capacities.append(shm.capacity // mini_batch_size)
        return {
            "capacities": capacities,
            "per_sample_capacities": per_sample_capacities,
        }

    def reader_meta(self, name=None):
        """Returns provided reader metadata as a dictionary. If no name is provided if provides
        a dictionary with data for all readers as {reader_name : meta}

        Available metadata keys:

        ``epoch_size``:        raw epoch size

        ``epoch_size_padded``: epoch size with the padding at the end to be divisible by
          the number of shards

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
        raise RuntimeError(
            "Current Pipeline not set!\n" + op_name
            + " operator must be used inside `define_graph` or "
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
        """Marks an edge as a data sink, preventing it from being pruned, even if it's not
        connected to the pipeline output.
        """
        self._sinks.append(edge)

    def _set_api_type(self, type):
        if type not in types.PipelineAPIType:
            raise RuntimeError(
                "Wrong pipeline API set!"
                "check available values in :meth:`nvidia.dali.types.PipelineAPIType`")
        self._api_type = type

    def _check_api_type(self, type):
        if self._api_type is None:
            self._set_api_type(type)
        if type != self._api_type:
            raise RuntimeError(f"Mixing pipeline API type. Currently used: {self._api_type}, "
                               f"but trying to use {str(type)}")

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
                raise RuntimeError(
                    "Duplicate graph definition - `define_graph` argument "
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
                    raise TypeError(
                        f"Illegal output type. The output {i} is a `{type(outputs[i])}`. "
                        f"Allowed types are ``DataNode`` and types convertible to "
                        f"`types.Constant` (numerical constants, 1D lists/tuple of numbers "
                        f"and ND arrays).")

            _data_node._check(outputs[i])

        self._ops = _collect_ops(list(outputs) + self._sinks)
        self._graph_outputs = outputs
        self._setup_input_callbacks()
        self._disable_pruned_external_source_instances()
        self._py_graph_built = True

    def _setup_pipe_pool_dependency(self):
        if self._py_pool_started:
            # The sole point of this call is to ensure the lifetime of the pool exceeds the lifetime
            # of the pipeline's backend, so that shared memory managed by the pool is not freed
            # before pipline's backend is garbage collected.
            # Otherwise the backend may try to access unmmaped memory which leads to
            # crashes at the Python teardown.
            self._pipe.SetPyObjDependency(self._py_pool)

    def _start_py_workers(self):
        if not self._parallel_input_callbacks:
            return
        self._py_pool = WorkerPool.from_groups(self._parallel_input_callbacks,
                                               self._prefetch_queue_depth,
                                               self._max_batch_size,
                                               self._py_start_method,
                                               self._py_num_workers,
                                               py_callback_pickler=self._py_callback_pickler)
        # ensure processes started by the pool are termineted when pipeline is no longer used
        weakref.finalize(self, lambda pool: pool.close(), self._py_pool)
        self._py_pool_started = True

    def _init_pipeline_backend(self):
        device_id = self._device_id if self._device_id is not None else types.CPU_ONLY_DEVICE_ID
        if device_id != types.CPU_ONLY_DEVICE_ID:
            b.check_cuda_runtime()
        self._pipe = b.Pipeline(self._max_batch_size,
                                self._num_threads,
                                device_id,
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

    def _disable_pruned_external_source_instances(self):
        def truncate_str(obj, max_len=103):
            obj_str = str(obj)
            if len(obj_str) <= max_len:
                return obj_str
            return obj_str[:max_len - 3] + "..."

        graph_op_ids = set(op.id for op in self._ops)
        for group in self._input_callbacks:
            pruned_mask = [op.id not in graph_op_ids for op in group.instances]
            if any(pruned_mask):
                group.disable_pruned_instances(pruned_mask)
                pruned_idx = [i for i, is_pruned in enumerate(pruned_mask) if is_pruned]
                source_str = truncate_str(group.source_desc.source)
                num_outputs = len(group.instances)
                pruned_idx_str = ", ".join(str(idx) for idx in pruned_idx)
                if len(pruned_idx) > 1:
                    pruned_str = f"outputs at the indices {pruned_idx_str} are"
                else:
                    pruned_str = f"output at the index {pruned_idx_str} is"
                warnings.warn(
                    f"The external source node '{source_str}' produces {num_outputs} outputs, "
                    f"but the {pruned_str} not used. For best performance, adjust your "
                    f"callback so that it computes only the needed outputs.",
                    Warning
                )

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
            parallel = [group for group in groups if group.parallel]
            dedicated_worker_cbs = [
                group for group in parallel if WorkerPool.is_iterable_group(group)
            ]
            general_cbs = [group for group in parallel if not WorkerPool.is_iterable_group(group)]
            # make the callbacks that need dedicated worker first in line for prefetching, so that
            # the worker doesn't get busy with other tasks when dedicated tasks arrive
            self._parallel_input_callbacks = dedicated_worker_cbs + general_cbs
            self._seq_input_callbacks = [group for group in groups if not group.parallel]

    def start_py_workers(self):
        """
        Start Python workers (that will run ``ExternalSource`` callbacks).
        You need to call :meth:`start_py_workers` before you call any functionality that creates
        or acquires CUDA context when using ``fork`` to start Python
        workers (``py_start_method="fork"``). It is called automatically by
        :meth:`Pipeline.build` method when such separation is not necessary.

        If you are going to build more than one pipeline that starts Python workers by forking
        the process then you need to call :meth:`start_py_workers` method on all those pipelines
        before calling :meth:`build` method of any pipeline, as build acquires CUDA context
        for current process.

        The same applies to using any other functionality that would create CUDA context -
        for example, initializing a framework that uses CUDA or creating CUDA tensors with it.
        You need to call :meth:`start_py_workers` before you call such functionality when
        using ``py_start_method="fork"``.

        Forking a process that has a CUDA context is unsupported and may lead to unexpected errors.

        If you use the method you cannot specify ``define_graph`` argument
        when calling :meth:`build`.
        """
        if not self._py_graph_built:
            self._build_graph()
        if not self._py_pool_started:
            self._start_py_workers()

    def build(self):
        """Build the pipeline.

        Pipeline needs to be built in order to run it standalone.
        Framework-specific plugins handle this step automatically.
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

        self._pipe.Build(self._generate_build_args())
        self._built = True

    def _feed_input(self, name, data, layout=None, cuda_stream=None, use_copy_kernel=False):
        from nvidia.dali.external_source import _prep_data_for_feed_input
        if cuda_stream is None:
            cuda_stream = types._get_default_stream_for_array(data)
        if cuda_stream == -1:
            cuda_stream = None
        else:
            cuda_stream = types._raw_cuda_stream(cuda_stream)

        data = _prep_data_for_feed_input(data, self._max_batch_size, layout, self._device_id)

        if isinstance(data, list):
            self._pipe.SetExternalTensorInput(name, data, ctypes.c_void_p(cuda_stream),
                                              use_copy_kernel)
        else:
            self._pipe.SetExternalTLInput(name, data, ctypes.c_void_p(cuda_stream), use_copy_kernel)

    def feed_input(self, data_node, data, layout=None, cuda_stream=None, use_copy_kernel=False):
        """Pass a multidimensional array or DLPack (or a list thereof) to an eligible operator.

        The operators that may be provided with data using this function are the input operators
        (i.e. everything in ``fn.inputs`` module) and the :meth:`fn.external_source`.

        In the case of the GPU input, the data must be modified on the same stream as the one
        used by ``feed_input``. See ``cuda_stream`` parameter for details.

        Parameters
        ----------
        data_node : :class:`DataNode` or a string
            The name of an eligible operator node or a :class:`DataNode`
            object returned by a call to that operator.

        data : ndarray or DLPack or a list thereof
            The array(s) may be one of:

              * NumPy ndarray (CPU)
              * MXNet ndarray (CPU)
              * PyTorch tensor (CPU or GPU)
              * CuPy array (GPU)
              * objects implementing ``__cuda_array_interface__``
              * DALI ``TensorList`` or list of DALI ``Tensor`` objects

            The data to be used as the output of the operator referred to by ``data_node``.

        layout : string or ``None``
            The description of the data layout (or empty string, if not specified).
            It should be a string of the length that matches the dimensionality of the data, batch
            dimension excluded. For a batch of channel-first images, this should be ``"CHW"``, for
            channel-last video it's ``"FHWC"`` and so on.
            If ``data`` is a DALI ``TensorList`` or a list of DALI ``Tensor`` objects and ``layout``
            is ``None``, the layout is taken from ``data``.
            The layout of the data must be the same in each iteration.

        cuda_stream : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``,
            e.g. ``cupy.cuda.Stream``, ``torch.cuda.Stream``
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

        use_copy_kernel : optional, ``bool``
            If set to True, DALI will use a CUDA kernel to feed the data (only applicable
            when copying data to/from GPU memory) instead of ``cudaMemcpyAsync`` (default).
        """
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if isinstance(data_node, str):
            name = data_node
        else:
            _data_node._check(data_node)
            name = data_node.name

        # Check for use of feed_input on an external_source operator that was
        # initialized with 'source'. This check makes sense only for fully Python-based
        # pipelines, and not deserialized ones.
        from .external_source import _is_external_source
        if not self._deserialized:
            if next(
                    (_is_external_source(op) and op._callback is not None
                     for op in self._ops if op.name == name),
                    False):
                raise RuntimeError(
                    f"Cannot use `feed_input` on the external source '{name}' with a `source`"
                    " argument specified.")

        self._feed_input(name, data, layout, cuda_stream, use_copy_kernel)

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

    def _are_pipeline_inputs_possible(self):
        """
        Returns True if using pipeline_inputs argument in .run() function is possible.
        """
        if not self.exec_pipelined:
            return True
        if self.exec_separated:
            return self._cpu_queue_size <= 1 and self._gpu_queue_size <= 1
        return self.prefetch_queue_depth <= 1

    def run(self, **pipeline_inputs):
        """
        Run the pipeline and return the result.

        If the pipeline was created with `exec_pipelined` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution.
        Should not be mixed with :meth:`schedule_run` in the same pipeline,
        :meth:`share_outputs` and
        :meth:`release_outputs`

        Parameters
        ----------
        pipeline_inputs :
            Optional argument that can be used to provide inputs to DALI.
            When DALI has any input operators defined (e.g. fn.external_source), you can provide the
            inputs to those using named arguments in this function. The assumption is that
            DALI pipeline has them defined and named properly::

                @pipeline_def
                def my_pipe():
                    inp = fn.external_source(name="my_inp")
                    return inp

            With the example pipeline above, you can provide ``"my_inp"`` input into the
            :meth:`run()` function::

                p = my_pipe(prefetch_queue_depth=1, ...)
                p.build()
                p.run(my_inp=np.random((2,3,2)))

            Such keyword argument specified in the :meth:`run()` function has to have a
            corresponding input operator node declared in DALI pipeline.

            As always when working with DALI, the value passed to the keyword argument has to
            denote a whole batch of data.

            Please note, that using this feature requires setting either ``prefetch_queue_depth=1``
            or ``exec_pipelined=False`` in DALI Pipeline constructor.

            This feature can be considered as a syntactic sugar over :meth:`feed_input` function.

        Returns
        -------
            A list of `TensorList` objects for respective pipeline outputs
        """
        if len(pipeline_inputs) > 0 and not self._are_pipeline_inputs_possible():
            raise RuntimeError(f"""
                When using pipeline_inputs named arguments, either
                `prefetch_queue_depth` in Pipeline constructor shall be set to 1 (for both devices)
                or `exec_pipelined` shall be set to False.
                Received: prefetch_queue_depth={self.prefetch_queue_depth},
                exec_pipelined={self.exec_pipelined}.
                Please set the `prefetch_queue_depth` or `exec_pipelined` argument in the Pipeline
                constructor properly or provide inputs to DALI Pipeline via another mean
                (e.g. `feed_input` function or `source` argument in the `fn.external_source`
                operator.)""")
        for inp_name, inp_value in pipeline_inputs.items():
            self.feed_input(inp_name, inp_value)
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
            group.prefetch(self._py_pool, i, self._max_batch_size, self._epoch_idx)

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
            self._epoch_idx += 1
            if self._input_callbacks:
                for group in self._input_callbacks:
                    group.reset_indices()
                for i, group in enumerate(self._parallel_input_callbacks):
                    # iterators are not reset or their prefetch results discarded
                    # unless they have caused an exception
                    if not self._py_pool.is_iterable_group(group):
                        self._py_pool.reset_context(i)

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
            raise TypeError("Provided `define_graph` argument is not callable."
                            + (" Didn't you want to write `.serialize(filename=...)`?"
                               if isinstance(define_graph, str) else ""))
        if not self._py_graph_built:
            self._build_graph(define_graph)
        if not self._backend_prepared:
            self._init_pipeline_backend()
            self._pipe.SetOutputDescs(self._generate_build_args())
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
            raise ValueError("serialized_pipeline and filename arguments are mutually exclusive. "
                             "Precisely one of them should be defined.")
        pipeline = cls()
        if filename is not None:
            with open(filename, 'rb') as pipeline_file:
                serialized_pipeline = pipeline_file.read()
        pipeline._pipe = b.Pipeline(serialized_pipeline,
                                    kw.get("batch_size", -1),
                                    kw.get("num_threads", -1),
                                    kw.get("device_id", -1),
                                    kw.get("exec_pipelined", True),
                                    kw.get("prefetch_queue_depth", 2),
                                    kw.get("exec_async", True),
                                    kw.get("bytes_per_sample", 0),
                                    kw.get("set_affinity", False),
                                    kw.get("max_streams", -1),
                                    kw.get("default_cuda_stream_priority", 0))
        if pipeline.device_id != types.CPU_ONLY_DEVICE_ID:
            b.check_cuda_runtime()
        pipeline._pipe.SetExecutionTypes(pipeline._exec_pipelined, pipeline._exec_separated,
                                         pipeline._exec_async)
        pipeline._pipe.SetQueueSizes(pipeline._cpu_queue_size, pipeline._gpu_queue_size)
        pipeline._pipe.EnableExecutorMemoryStats(pipeline._enable_memory_stats)
        pipeline._backend_prepared = True
        pipeline._pipe.Build()
        pipeline._built = True
        pipeline._deserialized = True
        pipeline._max_batch_size = kw.get("batch_size", -1)
        pipeline._num_threads = kw.get("num_threads", -1)
        pipeline._device_id = kw.get("device_id", -1)
        pipeline._exec_pipelined = kw.get("exec_pipelined", True)
        pipeline._prefetch_queue_depth = kw.get("prefetch_queue_depth", 2)
        pipeline._exec_async = kw.get("exec_async", True)
        pipeline._bytes_per_sample = kw.get("bytes_per_sample", 0)
        pipeline._set_affinity = kw.get("set_affinity", False)
        pipeline._max_streams = kw.get("max_streams", -1)
        pipeline._default_cuda_stream_priority = kw.get("default_cuda_stream_priority", 0)

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
        self._deserialized = True

    def save_graph_to_dot_file(self, filename, show_tensors=False, show_ids=False,
                               use_colors=False):
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

        batches = []  # data from external source callbacks is gathered here
        stop_iter = False
        for i, group in enumerate(self._parallel_input_callbacks):
            try:
                batches.append(
                    group.schedule_and_receive(self, self._py_pool, i, self._max_batch_size,
                                               self._epoch_idx))
            except StopIteration:
                stop_iter = True
        for group in self._seq_input_callbacks:
            try:
                batches.append(group.get_batch(self, self._max_batch_size, self._epoch_idx))
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

    def _generate_build_args(self):
        num_outputs = len(self._names_and_devices)
        dtypes = [self._output_dtype] * num_outputs if type(
            self._output_dtype) is not list else self._output_dtype
        ndims = [self._output_ndim] * num_outputs if type(
            self._output_ndim) is not list else self._output_ndim
        if not (len(dtypes) == len(ndims) == num_outputs):
            raise RuntimeError(
                f"Lengths of provided output descriptions do not match. \n"
                f"Expected num_outputs={num_outputs}."
                f"\nReceived:\noutput_dtype={dtypes}\noutput_ndim={ndims}"
            )

        return [(name, dev, types.NO_TYPE if dtype is None else dtype, -1 if ndim is None else ndim)
                for (name, dev), dtype, ndim in zip(self._names_and_devices, dtypes, ndims)]


def _discriminate_args(func, **func_kwargs):
    """Split args on those applicable to Pipeline constructor and the decorated function."""
    func_argspec = inspect.getfullargspec(func)
    ctor_argspec = inspect.getfullargspec(Pipeline.__init__)

    if 'debug' not in func_argspec.args and 'debug' not in func_argspec.kwonlyargs:
        func_kwargs.pop('debug', False)

    if ('enable_conditionals' not in func_argspec.args
            and 'enable_conditionals' not in func_argspec.kwonlyargs):
        func_kwargs.pop('enable_conditionals', False)

    ctor_args = {}
    fn_args = {}

    if func_argspec.varkw is not None:
        raise TypeError(
            f"Using variadic keyword argument `**{func_argspec.varkw}` in a  "
            f"graph-defining function is not allowed.")

    for farg in func_kwargs.items():
        is_ctor_arg = farg[0] in ctor_argspec.args or farg[0] in ctor_argspec.kwonlyargs
        is_fn_arg = farg[0] in func_argspec.args or farg[0] in func_argspec.kwonlyargs
        if is_fn_arg:
            fn_args[farg[0]] = farg[1]
            if is_ctor_arg:
                print(
                    f"Warning: the argument `{farg[0]}` shadows a Pipeline constructor "
                    "argument of the same name.")
        elif is_ctor_arg:
            ctor_args[farg[0]] = farg[1]
        else:
            assert False, f"This shouldn't happen. Please double-check the `{farg[0]}` argument"

    return ctor_args, fn_args


def _regroup_args(func, pipeline_def_kwargs, fn_call_kwargs):
    """Regroup arguments that are directed into Pipeline object construction (Pipeline kwargs)
    and those that are passed into pipeline definition function (Function kwargs).

    Parameters
    ----------
    func : Callable
        The pipeline definition function that is decorated.
    pipeline_def_kwargs : Dict
        Kwargs passed to the @pipeline_def
    fn_call_kwargs : Dict
        Kwargs passed when invoking the decorated function

    Returns
    -------
    (Dict, Dict)
        Pipeline kwargs, Function kwargs
    """
    ctor_args, fn_kwargs = _discriminate_args(func, **fn_call_kwargs)
    pipeline_kwargs = {**pipeline_def_kwargs, **ctor_args}  # Merge and overwrite dict
    return pipeline_kwargs, fn_kwargs


def _preprocess_pipe_func(func, conditionals_on):
    """Transform the pipeline definition function if the conditionals are enabled
    """
    if conditionals_on:
        return _conditionals._autograph.to_graph(func)
    else:
        return func


def _preprocess_pipe_object(pipe, conditionals_on, args, fn_kwargs):
    """Based on the conditional mode status, preprocess the pipeline object before the graph
    is created.
    """
    if conditionals_on:
        # We push and pop manually to be compatible with _PipelineDebug
        try:
            Pipeline.push_current(pipe)
            pipe._conditionals_enabled = True
            pipe._condition_stack = _conditionals._ConditionStack()
            # Add all parameters to the pipeline as "know" nodes in the top scope.
            for arg in args:
                if isinstance(arg, DataNode):
                    _conditionals.register_data_nodes(arg)
            for _, arg in fn_kwargs.items():
                if isinstance(arg, DataNode):
                    _conditionals.register_data_nodes(arg)
        finally:
            Pipeline.pop_current()


def _generate_graph(pipe, func, fn_args, fn_kwargs):
    """Build the graph provided by pipeline definition in `func` within the `pipe`.

    Parameters
    ----------
    pipe : Pipeline
        Target pipeline object
    func : Callable
        The pipeline definition that is decorated
    fn_args : List
        Positional arguments to `func`
    fn_kwargs : Dict
        Kwargs to `func`
    """
    with pipe:
        pipe_outputs = func(*fn_args, **fn_kwargs)
        if isinstance(pipe_outputs, tuple):
            po = pipe_outputs
        elif pipe_outputs is None:
            po = ()
        else:
            po = (pipe_outputs, )
        pipe.set_outputs(*po)


def pipeline_def(fn=None, *, enable_conditionals=False, **pipeline_kwargs):
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
    GPU device id and so on (see :meth:`nvidia.dali.Pipeline()` for a
    complete list of pipeline parameters).
    These parameters can be supplied as additional keyword arguments,
    passed to the decorated function::

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

    Keyword args
    ------------

    enable_conditionals : bool, optional
        Enable support for conditional execution of DALI operators using ``if`` statements
        in the pipeline definition, by default False.
    """

    def actual_decorator(func):

        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            conditionals_on = kwargs.get('enable_conditionals', enable_conditionals)

            pipe_func = _preprocess_pipe_func(func, conditionals_on)
            pipeline_args, fn_kwargs = _regroup_args(pipe_func, pipeline_kwargs, kwargs)
            pipe = Pipeline(**pipeline_args)
            _preprocess_pipe_object(pipe, conditionals_on, args, fn_kwargs)

            _generate_graph(pipe, pipe_func, args, fn_kwargs)
            return pipe

        # Add `is_pipeline_def` attribute to the function marked as `@pipeline_def`
        create_pipeline._is_pipeline_def = True
        return create_pipeline

    return actual_decorator(fn) if fn else actual_decorator


def _collect_ops(output_nodes):
    """
    Traverses the pipeline graph starting from the outputs to collect all reachable operators.
    Returns the list of operators topologically sorted, so that operators that contribute
    as inputs to another operator go first.
    """

    def get_source_op(edge: DataNode):
        source_op = edge.source
        if source_op is None:
            raise RuntimeError("Pipeline encountered an Edge with no source op.")
        return source_op

    def get_op_input_edges(op) -> List[DataNode]:
        for inp in op.inputs:
            if isinstance(inp, list):
                yield from inp
            else:
                yield inp

    def get_op_outputs_num():
        # BSF traverse the graph first to learn, for each reachable operator in the graph,
        # how many data-nodes/edges the operator contributes to
        # (i.e. the number of outputs of the operator instance)
        op_outputs_num = {}
        edges = deque(output_nodes)
        while edges:
            current_edge = edges.popleft()
            source_op = get_source_op(current_edge)
            if source_op.id in op_outputs_num:
                op_outputs_num[source_op.id] += 1
            else:
                op_outputs_num[source_op.id] = 1
                source_op.check_args()
                edges.extend(get_op_input_edges(source_op))
        return op_outputs_num

    ops = []
    edges = deque(output_nodes)
    op_total_outputs_num = get_op_outputs_num()
    op_visited_outputs_num = {op_id: 0 for op_id in op_total_outputs_num}
    while edges:
        current_edge = edges.popleft()
        source_op = get_source_op(current_edge)
        op_visited_outputs_num[source_op.id] += 1
        # Actually visit the operator only when all the nodes it contributes to
        # were already processed
        if op_visited_outputs_num[source_op.id] == op_total_outputs_num[source_op.id]:
            ops.append(source_op)
            edges.extend(get_op_input_edges(source_op))
    ops.reverse()
    return ops


def _pipeline_def_experimental(fn=None, *, enable_conditionals=False, **pipeline_kwargs):
    """Variant of :meth:`@pipeline_def <nvidia.dali.pipeline_def>` decorator that enables additional
    experimental features. It has the same API as its non-experimental variant with the addition of
    the keyword arguments listed below.

    Keyword args
    ------------
    debug : bool, optional
        Enable pipeline debug mode - allowing for step-by-step execution and intermediate data
        inspection of the pipeline definition, by default False.

        .. note::
            This mode is intended only for debugging purposes - the pipeline performance will be
            significantly worse than the non-debug mode.

    .. note::

        The features enabled by this decorator are experimental. The API may change and the
        functionality may be limited.
    """
    from nvidia.dali._debug_mode import _PipelineDebug
    pipeline_debug = pipeline_kwargs.pop('debug', False)

    def actual_decorator(func):

        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            debug_mode_on = kwargs.get('debug', pipeline_debug)
            conditionals_on = kwargs.get('enable_conditionals', enable_conditionals)

            pipe_func = _preprocess_pipe_func(func, conditionals_on)
            pipeline_args, fn_kwargs = _regroup_args(pipe_func, pipeline_kwargs, kwargs)
            if debug_mode_on:
                pipe = _PipelineDebug(functools.partial(pipe_func, *args, **fn_kwargs),
                                      **pipeline_args)
            else:
                pipe = Pipeline(**pipeline_args)

            _preprocess_pipe_object(pipe, conditionals_on, args, fn_kwargs)

            if not debug_mode_on:
                _generate_graph(pipe, pipe_func, args, fn_kwargs)

            return pipe

        # Add `is_pipeline_def` attribute to the function marked as `@pipeline_def`
        create_pipeline._is_pipeline_def = True
        return create_pipeline

    return actual_decorator(fn) if fn else actual_decorator


def _insert_experimental_pipeline_def():
    current_module = sys.modules[__name__]
    experimental_module = internal.get_submodule(current_module, 'experimental')
    _pipeline_def_experimental.__module__ = experimental_module
    setattr(experimental_module, 'pipeline_def', _pipeline_def_experimental)


_insert_experimental_pipeline_def()
