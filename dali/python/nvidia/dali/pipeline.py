# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, List, Tuple, Callable, Optional, Union, TypeVar, overload
from nvidia.dali import backend as b
from nvidia.dali import types
from nvidia.dali import internal
from nvidia.dali import tensors
from nvidia.dali._multiproc.pool import WorkerPool
from nvidia.dali import pickling as dali_pickle
from nvidia.dali import _conditionals
from nvidia.dali._utils import dali_trace as _dali_trace
from nvidia.dali._utils.external_source_impl import SourceKind as _SourceKind
from threading import local as tls
from . import data_node as _data_node
import atexit
import copy
import functools
import inspect
import pickle  # nosec B403
import sys
import traceback
import warnings
import weakref
from .data_node import DataNode
from enum import Enum

pipeline_tls = tls()

DataNode.__module__ = __name__  # move to pipeline


def _show_deprecation_warning(deprecated, in_favor_of):
    # show only this warning
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.warn(
            "{} is deprecated, please use {} instead".format(deprecated, in_favor_of),
            Warning,
            stacklevel=2,
        )


def _show_warning(message):
    warnings.warn(message, Warning, stacklevel=2)


class StreamPolicy(Enum):
    """Stream policy for the pipeline.

    SINGLE:
        All operators will share a single stream.
    PER_BACKEND:
        Each backend will have its own stream.
    PER_OPERATOR:
        The operators that can run in parallel will have distinct streams.
        The number of streams is kept at minimum required for independent scheduling - for example
        strictly sequential pipelines will have only one stream.
    """

    SINGLE = b._ExecutorFlags.StreamPolicySingle
    PER_BACKEND = b._ExecutorFlags.StreamPolicyPerBackend
    PER_OPERATOR = b._ExecutorFlags.StreamPolicyPerOperator
    UNSPECIFIED = b._ExecutorFlags.NoFlags


class OperatorConcurrency(Enum):
    """Operator concurrency policy for the pipeline.

    NONE:
        No concurrency.
    BACKEND:
        Independent operators with different backends (cpu, gpu, mixed) will run in parallel.
    FULL:
        All independent operators will run in parallel.
        NOTE: Due to internal limitations, CPU operators cannot run in paralle with each other.
    """

    NONE = b._ExecutorFlags.ConcurrencyNone
    BACKEND = b._ExecutorFlags.ConcurrencyBackend
    FULL = b._ExecutorFlags.ConcurrencyFull
    UNSPECIFIED = b._ExecutorFlags.NoFlags


class Pipeline(object):
    """Pipeline class is the base of all DALI data pipelines. The pipeline
    encapsulates the data processing graph and the execution engine.

    Parameters
    ----------
    batch_size : int
        Maximum batch size of the pipeline. The value must be positive.
        In most cases, the actual batch size of the pipeline
        will be equal to the maximum one. Running the DALI Pipeline with a smaller batch size
        is also supported. The batch size might change from iteration to iteration.

        Please note, that DALI might perform memory preallocations according to this
        parameter. Setting it too high might result in out-of-memory failure.
    num_threads : int, optional, default = -1
        Number of CPU threads used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    device_id : int, optional, default = None
        Id of GPU used by the pipeline.
        If the pipeline requires a GPU and this field is left unspecified, DALI will use the
        current CUDA device (according to ``cudaGetDevice``) at the time the pipeline is built.
    seed : int, optional, default = None
        Seed used for random number generation. If not set, a random seed will be generated.
    exec_pipelined : bool, optional, default = True
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    prefetch_queue_depth : int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
        Depth of the executor pipeline. Deeper pipeline makes DALI
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:

        ``{ "cpu_size": x, "gpu_size": y }``

        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages.
        Executor will buffer cpu and gpu stages separately,
        and will fill the buffer queues when the first :meth:`run`
        is issued.
        Separated execution requires that ``exec_async=True``, ``exec_pipelined=True`` and
        ``exec_dynamic=False``.
    exec_async : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`run` method
        run asynchronously with respect to the calling Python thread.
        In order to synchronize with the pipeline one needs to call
        :meth:`outputs` method.
    exec_dynamic : bool, optional, default = None
        Whether to use the dynamic executor.
        Dynamic executor allows to interleave CPU and GPU operators and to perform GPU to CPU
        copies. It also uses dynamic memory allocation for pipeline outputs and inter-operator
        buffers, which reduces memory consumption in complex pipelines.
        The dynamic executor is used by default when `exec_async` and `exec_pipelined` are ``True``
        and separated queues are not used (see `prefetch_queue_depth`). It can be forcibly disabled
        by specifying ``False``.
    stream_policy : StreamPolicy, optional, default = None
        Stream policy (only for dynamic executor).
        If not specified, the default value is ``StreamPolicy.PER_BACKEND``.
    concurrency : OperatorConcurrency, optional, default = None
        Operator concurrency policy (only for dynamic executor).
        If not specified, the default value is ``OperatorConcurrency.BACKEND``.
    bytes_per_sample : int, optional, default = 0
        A hint for DALI for how much memory to use for its tensors.
    set_affinity : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    max_streams : int, deprecated, default = None
        Deprecated, this parameter has no effect.
    default_cuda_stream_priority : int, optional, default = None
        Deprecated, this parameter has no effect.
    enable_memory_stats : bool, optional, default = False
        If DALI should print operator output buffer statistics.
        Useful for `bytes_per_sample_hint` operator parameter.
        This flag has no effect when `exec_dynamic` is ``True``.
    enable_checkpointing : bool, optional, default = False
        If True, DALI will trace states of the operators. In that case, calling the `checkpoint`
        method returns serialized state of the pipeline. The same pipeline can be later rebuilt
        with the serialized state passed as the `checkpoint` parameter to resume running
        from the saved iteration.

        More details can be found in
        `this documentation section <advanced_topics_checkpointing.html>`_.
    checkpoint : str, optional, default = None
        Serialized checkpoint, received from `checkpoint` method.
        When pipeline is built, its state is restored from the `checkpoint` and the pipeline
        resumes execution from the saved iteration.

        More details can be found in
        `this documentation section <advanced_topics_checkpointing.html>`_.
    py_num_workers : int, optional, default = 1
        The number of Python workers that will process parallel
        :meth:`~nvidia.dali.fn.external_source` callbacks.
        The pool starts only if there is at least one ExternalSource with
        :paramref:`~nvidia.dali.fn.external_source.parallel` set to True.
        Setting it to 0 disables the pool and all ExternalSource operators fall back to non-parallel
        mode even if :paramref:`~nvidia.dali.fn.external_source.parallel` is set to True.
    py_start_method : str, default = "fork"
        Determines how Python workers are started. Supported methods:

          * ``"fork"`` - start by forking the process
          * ``"spawn"`` - start a fresh interpreter process

        If ``spawn`` method is used, ExternalSource's callback must be picklable.
        In order to use ``fork``, there must be no CUDA contexts acquired at the moment of starting
        the workers. For this reason, if you need to build multiple pipelines that use Python
        workers, you will need to call :meth:`start_py_workers` before building or running
        any of the pipelines (see :meth:`build` for details). You can find more details and caveats
        of both methods in Python's ``multiprocessing`` module documentation.
    py_callback_pickler : module or tuple, default = None
        If `py_start_method` is set to *spawn*, callback passed to parallel
        ExternalSource must be picklable.
        If run in Python3.8 or newer with `py_callback_pickler` set to None, DALI uses customized
        pickle when serializing callbacks to support serialization of local functions and lambdas.

        However, if you need to serialize more complex objects like local classes or you are running
        older version of Python you can provide external serialization package such as dill or
        cloudpickle that implements two methods: `dumps` and `loads` to make DALI use them to
        serialize external source callbacks. You can pass a module directly as
        `py_callback_pickler`::

            import dill
            @pipeline_def(py_callback_pickler=dill, ...)
            def create_pipeline():
                src = fn.external_source(
                    lambda sample_info: np.int32([42]),
                    batch=False,
                    parallel=True,
                )
                ...

        A valid value for `py_callback_pickler` is either a module/object implementing
        ``dumps`` and ``loads`` methods or a tuple where the first item is the module/object and
        the next two optional parameters are extra kwargs to be passed when calling dumps and
        loads respectively.
        The provided methods and kwargs must be picklable with standard `pickle.dumps`.

        If you run Python3.8 or newer with the default DALI pickler (`py_callback_pickler` = None),
        you can hint DALI to serialize global functions by value rather than by reference
        by decorating them with `@dali.pickling.pickle_by_value`. It may be especially useful when
        working with Jupyter notebook to work around the issue of worker process being unable to
        import the callback defined as a global function inside the notebook.
    output_dtype : ``nvidia.dali.types.DALIDataType`` or list of those, default = None
        With this argument, you may declare, what data type you expect in the given output. You
        shall pass a list of mod:`types.DALIDataType`, each element in the list corresponding to
        one output from the pipeline. Additionally, you can pass ``None`` as a wildcard.
        The outputs, after each iteration, will be validated against the types you passed to this
        argument. If any output does not match the provided type, RuntimeError will be raised.

        If the `output_dtype` value is a single value (not a list), it will be broadcast to the
        number of outputs from the pipeline.
    output_ndim : int or list of ints, default = None
        With this argument, you may declare, how many dimensions you expect in the given output.
        You shall pass a list of integers, each element in the list corresponding to one output
        from the pipeline.
        Additionally, you can pass ``None`` as a wildcard. The outputs, after each iteration, will
        be validated against the numbers of dimensions you passed to this argument. If the
        dimensionality of any output does not match the provided ``ndim``, RuntimeError will be
        raised.

        If the `output_ndim` value is a single value (not a list), it will be broadcast to the
        number of outputs from the pipeline."""

    def __init__(
        self,
        batch_size=None,
        num_threads=None,
        device_id=None,
        seed=None,
        exec_pipelined=True,
        prefetch_queue_depth=2,
        exec_async=True,
        bytes_per_sample=0,
        set_affinity=False,
        max_streams=None,
        default_cuda_stream_priority=None,
        *,
        enable_memory_stats=False,
        enable_checkpointing=False,
        checkpoint=None,
        py_num_workers=1,
        py_start_method="fork",
        py_callback_pickler=None,
        output_dtype=None,
        output_ndim=None,
        output_layout=None,
        exec_dynamic=None,
        experimental_exec_dynamic=None,
        stream_policy=None,
        concurrency=None,
    ):
        if experimental_exec_dynamic is not None:
            _show_deprecation_warning("experimental_exec_dynamic", "exec_dynamic")
            exec_dynamic = experimental_exec_dynamic
        if default_cuda_stream_priority is not None:
            _show_warning("The `default_cuda_stream_priority` is deprecated and has no effect.")
        if max_streams is not None:
            _show_warning("The `max_streams` is deprecated and has no effect.")
        self._pipe = None
        self._sinks = []
        self._max_batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        if seed is not None and seed < 0:
            warnings.warn(
                "The use of negative seed to use automatic seed asignment is deprecated. "
                "Use `None` instead."
            )
            seed = None
        self._seed = seed
        self._next_op_id_counter = 0
        self._exec_pipelined = exec_pipelined
        # When initializing DALI, we do the following in order:
        # * Discover the ops specified in Python, group the ExternalSources (_build_graph())
        # * Start the Python workers pool (_start_py_workers())
        # * Construct the C++ Pipeline backend and pass the graph to it (_init_pipeline_backend())
        # * Build the pipeline. (_pipe.Build())
        # In case of deserialized pipeline, only _backend_prepared and _built will be True
        self._py_graph_built = False
        self._py_pool_started = False
        self._backend_prepared = False
        self._built = False
        self._deserialized = False  # Marked True when deserializing
        self._first_iter = True
        self._last_iter = False
        self._epoch_idx = 0
        self._consumer_iter = 0
        self._consumer_epoch_idx = 0
        self._batches_to_consume = 0
        self._names_and_devices = None
        self._exec_async = exec_async
        self._exec_dynamic = exec_dynamic
        self._bytes_per_sample = bytes_per_sample
        self._set_affinity = set_affinity
        self._stream_policy = stream_policy
        self._concurrency = concurrency
        self._py_num_workers = py_num_workers
        self._py_start_method = py_start_method
        if py_callback_pickler is not None and py_start_method == "fork":
            raise ValueError(
                "``py_callback_pickler`` should not be set when 'fork' start method is used."
            )
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
        self._enable_checkpointing = enable_checkpointing
        self._checkpoint = checkpoint
        self._prefetch_queue_depth = prefetch_queue_depth
        self._is_restored_from_checkpoint = False
        self._iterator_data = None
        if type(prefetch_queue_depth) is dict:
            self._exec_separated = True
            if not exec_async:
                raise ValueError(
                    "`exec_async` must not evaluate to `False` when using separated queues."
                )
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
        # Tracking the stack frame where pipeline definition starts
        self._definition_frame_start = 0

        if exec_dynamic is None and exec_pipelined and exec_async and not self._exec_separated:
            self._exec_dynamic = exec_dynamic = True

        self._executor_type = b._MakeExecutorType(
            self._exec_pipelined, self._exec_async, self._exec_separated, self._exec_dynamic
        )

        self._executor_flags = b._ExecutorFlags.NoFlags
        if self._set_affinity:
            self._executor_flags |= b._ExecutorFlags.SetAffinity

        if self._stream_policy is not None:
            self._executor_flags &= ~b._ExecutorFlags.StreamPolicyMask
            self._executor_flags |= self._stream_policy.value
        if self._concurrency is not None:
            self._executor_flags &= ~b._ExecutorFlags.ConcurrencyMask
            self._executor_flags |= self._concurrency.value

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
                        f"`output_dtype` can't be a types.NO_TYPE. Found {dtype} in the list."
                    )
        elif not isinstance(output_dtype, (types.DALIDataType, type(None))):
            raise TypeError(
                f"`output_dtype` must be either: a value from nvidia.dali.types.DALIDataType, a "
                f"list of these or None. Found type: {type(output_dtype)}."
            )
        elif output_dtype == types.NO_TYPE:
            raise ValueError(
                f"`output_dtype` can't be a types.NO_TYPE. Found value: {output_dtype}"
            )
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
                        f"`output_ndim` must be non-negative. Found value {ndim} in the list."
                    )
        elif not isinstance(output_ndim, (int, type(None))):
            raise TypeError(
                f"`output_ndim` must be either: an int, a list of ints or None. "
                f"Found type: {type(output_ndim)}."
            )
        elif output_ndim is not None and output_ndim < 0:
            raise ValueError(f"`output_ndim` must be non-negative. Found value: {output_ndim}.")
        self._output_ndim = output_ndim

        # Assign and validate output_layout
        if isinstance(output_layout, (list, tuple)):
            for layout in output_layout:
                if not isinstance(layout, (str, type(None))):
                    raise TypeError(
                        f"`output_layout` must be either: a string, a list of strings or None. "
                        f"Found type {type(layout)} in the list."
                    )
        elif not isinstance(output_layout, (str, type(None))):
            raise TypeError(
                f"`output_layout` must be either: a string, a list of strings or None. "
                f"Found type: {type(output_layout)}."
            )
        self._output_layout = output_layout

        Pipeline._pipes.add(weakref.ref(self))

    _pipes = set()  # this is necessary for clean exit

    def __del__(self):
        self._shutdown()

    def _shutdown(self):
        if self._pipe:
            self._pipe.Shutdown()
            self._pipe = None
        Pipeline._pipes.discard(weakref.ref(self))

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
        """Id of the GPU used by the pipeline or None, if not set.

        If the pipeline requires a GPU but none was specified at construction, the current GPU
        (according to CUDA runtime) will be assigned once the pipeline is built.
        """
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
    def exec_dynamic(self):
        """If true, the dynamic executor is used."""
        return self._exec_dynamic

    @property
    def stream_policy(self):
        """Stream policy for the pipeline."""
        return self._stream_policy

    @property
    def concurrency(self):
        """Operator concurrency for the pipeline."""
        return self._concurrency

    @property
    def set_affinity(self):
        """If True, worker threads are bound to CPU cores."""
        return self._set_affinity

    @property
    def max_streams(self):
        """Deprecated, unused; returns -1."""
        return -1

    @property
    def prefetch_queue_depth(self):
        """Depth (or depths) of the prefetch queue, as specified in the ``__init__`` arguments."""
        return self._prefetch_queue_depth

    @property
    def default_cuda_stream_priority(self):
        """Deprecated; always 0."""
        return 0

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

    @property
    def is_restored_from_checkpoint(self):
        """If True, this pipeline was restored from checkpoint."""
        return self._is_restored_from_checkpoint

    @property
    def num_outputs(self) -> int:
        """
        Number of pipeline outputs.
        """
        self.build()
        # output_dtype is a list with the dtype for each output, so we can simply take the length
        return len(self._pipe.output_dtype())

    def output_dtype(self) -> list:
        """Data types expected at the outputs."""
        self.build()
        return [elem if elem != types.NO_TYPE else None for elem in self._pipe.output_dtype()]

    def output_ndim(self) -> list:
        """Number of dimensions expected at the outputs."""
        self.build()
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

        self.build()
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

        .. note::
            Executor statistics are available only when ``exec_dynamic=False``.
        """
        self.build()
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
                shm.capacity
                for context in self._py_pool.contexts
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
        self.build()
        if name is not None:
            return self._pipe.reader_meta(name)
        return self._pipe.reader_meta()

    @staticmethod
    def current():
        """Returns the instance of the current pipeline set by :meth:`push_current`."""
        return getattr(pipeline_tls, "current_pipeline", None)

    @staticmethod
    def _raise_pipeline_required(op_name):
        raise RuntimeError(
            "Current Pipeline not set!\n"
            + op_name
            + " operator must be used inside `define_graph` or "
            "current pipeline must be explicitly set using context manager (`with my_pipeline:`) "
            "or with a call to `Pipeline.push_current(my_pipeline)`."
        )

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
        if _dali_trace.is_tracing_enabled():
            stack_start = _dali_trace.get_stack_depth()

            call_context = traceback.extract_stack(limit=3)
            current_filename = call_context[-1].filename
            # Cases:
            # * in user code, we keep last user frame for reference:
            #   user_function():
            #       with pipe / Pipeline.push_current():
            #           <pipeline_definition>
            # * in DALI internals, we need to remove everything below _generate_graph:
            #   _generate_graph():
            #       with pipe: -> __enter__()
            #           push_current()
            #       pipeline_def()
            #

            if (
                len(call_context) > 2
                and call_context[-2].filename == current_filename
                and call_context[-2].name == "__enter__"
                and call_context[-3].filename == current_filename
                and call_context[-3].name == "_generate_graph"
            ):
                # We point below the pipeline_def invocation
                pipeline._definition_frame_start = stack_start - 2
            else:
                # Otherwise we are in user code
                if (
                    len(call_context) > 1
                    and call_context[-2].filename == current_filename
                    and call_context[-2].name == "__enter__"
                ):
                    pipeline._definition_frame_start = stack_start - 3
                else:
                    pipeline._definition_frame_start = stack_start - 2

        stack = getattr(pipeline_tls, "pipeline_stack", None)
        if stack is None:
            pipeline_tls.pipeline_stack = [prev]
        else:
            stack.append(prev)
        return prev

    @staticmethod
    def pop_current():
        """Restores previous pipeline as current. Complementary to :meth:`push_current`."""
        pipeline_tls.current_pipeline._definition_frame_start = 0
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
                "check available values in :meth:`nvidia.dali.types.PipelineAPIType`"
            )
        self._api_type = type

    def _check_api_type(self, type):
        if self._api_type is None:
            self._set_api_type(type)
        if type != self._api_type:
            raise RuntimeError(
                f"Mixing pipeline API type. Currently used: {self._api_type}, "
                f"but trying to use {str(type)}"
            )

    def enable_api_check(self, enable):
        """Allows to enable or disable API check in the runtime"""
        self._skip_api_check = not enable

    def _check_api_type_scope(self, type):
        """Checks the API currently used by pipeline and throws an error if it differs

        It helps preventing of mixing simple, iterator and scheduled based API for
        pipeline run. Disables further checks in its scope
        """
        if not self._skip_api_check:
            self._check_api_type(type)

        class api_checker:
            def __init__(self, pipe):
                self._pipe = pipe

            def __enter__(self):
                self._old_skip_api_check = self._pipe._skip_api_check
                self._pipe._skip_api_check = True

            def __exit__(self, type, value, traceback):
                self._pipe._skip_api_check = self._old_skip_api_check

        return api_checker(self)

    def _require_unique_names(self):
        ops_by_name = {}
        for op in self._ops:
            ops = ops_by_name.get(op.name, None)
            if ops is None:
                ops = ops_by_name[op.name] = []
            ops.append(op)
        duplicate = {}
        foreign = False
        for name, ops in ops_by_name.items():
            if len(ops) > 1:
                duplicate[name] = ops
                for op in ops:
                    if op.pipeline is not self:
                        foreign = True

        if duplicate:
            message = (
                f"The pipeline is invalid because it contains operators with non-unique names:\n"
                f"{duplicate}"
            )
            if foreign:
                message += (
                    "\nThe likely cause is that the pipeline contains a subgraph "
                    "instantiated while a different pipeline was set as the current "
                    "pipeline (e.g. inside another pipeline's graph definition function).\n"
                )
            raise RuntimeError(message)

    def _require_no_foreign_ops(self, message):
        foreign = []
        for op in self._ops:
            if op.pipeline is not self:
                foreign.append(op)
        if foreign:
            raise RuntimeError(
                f"{message} because it contains operator(s) "
                f"that were defined outside the pipeline scope:\n"
                f"{[o.name for o in foreign]}\n"
                f"All operators should be defined while the pipeline is set as the current "
                f"pipeline. This happens automatically when defining the pipeline in a "
                f"function decorated with `@pipeline_def`."
            )

    # Graph is constructed by backtracking from the output edges and the edges marked as sinks
    def _build_graph(self, define_graph=None):
        if define_graph is not None:
            if self._graph_out is not None:
                raise RuntimeError(
                    "Duplicate graph definition - `define_graph` argument "
                    "should not be specified when graph was defined with a call to `set_outputs`."
                )
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

                outputs[i] = nvidia.dali.ops._instantiate_constant_node(outputs[i], "cpu")
            elif contains_nested_datanode(outputs[i]):
                raise TypeError(
                    f"Illegal pipeline output type. The output {i} contains a nested "
                    "`DataNode`. Missing list/tuple expansion (*) is the likely cause."
                )
            elif not isinstance(outputs[i], DataNode):
                try:
                    outputs[i] = types.Constant(outputs[i], device="cpu")
                except TypeError:
                    raise TypeError(
                        f"Illegal output type. The output {i} is a `{type(outputs[i])}`. "
                        f"Allowed types are ``DataNode`` and types convertible to "
                        f"`types.Constant` (numerical constants, 1D lists/tuple of numbers "
                        f"and ND arrays)."
                    )

            _data_node._check(outputs[i])

        self._ops = _collect_ops(list(outputs) + self._sinks)
        self._require_unique_names()
        if self._enable_checkpointing:
            self._require_no_foreign_ops("The pipeline does not support checkpointing")

        self._graph_outputs = outputs
        self._num_outputs = len(self._graph_outputs)
        self._setup_input_callbacks()
        self._disable_pruned_external_source_instances()
        self._py_graph_built = True

    def _setup_pipe_pool_dependency(self):
        if self._py_pool_started:
            # The sole point of this call is to ensure the lifetime of the pool exceeds the lifetime
            # of the pipeline's backend, so that shared memory managed by the pool is not freed
            # before pipeline's backend is garbage collected.
            # Otherwise the backend may try to access unmmaped memory which leads to
            # crashes at the Python teardown.
            self._pipe.SetPyObjDependency(self._py_pool)

    def _start_py_workers(self):
        if not self._parallel_input_callbacks:
            return
        self._py_pool = WorkerPool.from_groups(
            self._parallel_input_callbacks,
            self._prefetch_queue_depth,
            self._max_batch_size,
            self._py_start_method,
            self._py_num_workers,
            py_callback_pickler=self._py_callback_pickler,
        )
        # ensure processes started by the pool are terminated when pipeline is no longer used
        weakref.finalize(self, lambda pool: pool.close(), self._py_pool)
        self._py_pool_started = True

    def _get_params(self):
        return b._PipelineParams(
            max_batch_size=self._max_batch_size,
            num_threads=self._num_threads,
            device_id=self._device_id,
            seed=self._seed,
            executor_type=self._executor_type,
            executor_flags=self._executor_flags,
            prefetch_queue_depths=(self._cpu_queue_size, self._gpu_queue_size),
            enable_checkpointing=self._enable_checkpointing,
            enable_memory_stats=self._enable_memory_stats,
            bytes_per_sample_hint=self._bytes_per_sample,
        )

    def _set_params(self, params):
        self._max_batch_size = params.max_batch_size
        self._num_threads = params.num_threads
        self._device_id = params.device_id
        self._seed = params.seed
        self._executor_type = params.executor_type
        self._executor_flags = params.executor_flags
        self._cpu_queue_size = params.prefetch_queue_depths[0]
        self._gpu_queue_size = params.prefetch_queue_depths[1]
        self._enable_checkpointing = params.enable_checkpointing
        self._enable_memory_stats = params.enable_memory_stats
        self._bytes_per_sample = params.bytes_per_sample_hint
        # reconsitute legacy flags
        self._exec_async = bool(params.executor_type & b._ExecutorType.AsyncFlag)
        self._exec_pipelined = bool(params.executor_type & b._ExecutorType.PipelinedFlag)
        self._exec_separated = bool(params.executor_type & b._ExecutorType.SeparatedFlag)
        self._exec_dynamic = bool(params.executor_type & b._ExecutorType.DynamicFlag)
        self._set_affinity = bool(params.executor_flags & b._ExecutorFlags.SetAffinity)
        self._stream_policy = StreamPolicy(
            params.executor_flags & b._ExecutorFlags.StreamPolicyMask
        )
        self._concurrency = OperatorConcurrency(
            params.executor_flags & b._ExecutorFlags.ConcurrencyMask
        )
        if self.exec_separated:
            self._prefetch_queue_depth = {"cpu": self._cpu_queue_size, "gpu": self._gpu_queue_size}
        else:
            assert self._cpu_queue_size == self._gpu_queue_size
            self._prefetch_queue_depth = self._cpu_queue_size

    def _init_pipeline_backend(self):
        params = self._get_params()
        self._pipe = b.Pipeline(params)
        if self._pipe.requires_gpu():
            b.check_cuda_runtime()

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
            return obj_str[: max_len - 3] + "..."

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
                    Warning,
                )

    def _check_checkpointing_support(self):
        if not self._enable_checkpointing:
            return

        for group in self._input_callbacks:
            kind = group.source_desc.kind
            has_inputs = group.source_desc.has_inputs
            checkpointing_supported = kind == _SourceKind.CALLABLE and has_inputs

            if not checkpointing_supported:
                reason = "with unsupported 'source'"
                if kind != _SourceKind.CALLABLE:
                    reason = f"with {kind} as a 'source'"
                elif not has_inputs:
                    reason = "with parameterless callable as a 'source'"

                warnings.warn(
                    "Checkpointing enabled in a pipeline with external source operator, "
                    f"{reason}. "
                    "DALI doesn't capture state of such 'source'. When loading the checkpoint, "
                    "the 'source' must be manually adjusted by the user to start from the "
                    "correct point, otherwise it will start from the beginning, "
                    "potentially leading to mismatch with other data sources."
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
        parallel = [group for group in groups if group.parallel]
        if parallel and (not isinstance(self._py_num_workers, int) or self._py_num_workers <= 0):
            raise RuntimeError(
                f"The pipeline contains `fn.external_source` with `parallel` argument specified "
                f"to True. However, the `py_num_workers` was set to `{self._py_num_workers}`. "
                f"The external source cannot run in parallel mode without Python workers pool. "
                f"Please specify the number of `py_num_workers` to a positive integer or set the "
                f"`parallel` parameter to False in external sources in the pipeline."
            )
        dedicated_worker_cbs = [group for group in parallel if WorkerPool.is_iterable_group(group)]
        general_cbs = [group for group in parallel if not WorkerPool.is_iterable_group(group)]
        # make the callbacks that need dedicated worker first in line for prefetching, so that
        # the worker doesn't get busy with other tasks when dedicated tasks arrive
        self._parallel_input_callbacks = dedicated_worker_cbs + general_cbs
        self._seq_input_callbacks = [group for group in groups if not group.parallel]

        self._check_checkpointing_support()

    def start_py_workers(self):
        """
        Start Python workers (that will run ``ExternalSource`` callbacks).
        You need to call :meth:`start_py_workers` before you call any functionality that creates
        or acquires CUDA context when using ``fork`` to start Python
        workers (``py_start_method="fork"``). It is called automatically by
        :meth:`Pipeline.build` method when such separation is not necessary.

        If you are going to build more than one pipeline that starts Python workers by forking
        the process then you need to call :meth:`start_py_workers` method on all those pipelines
        before calling any method that builds or runs the pipeline (see :meth:`build` for details),
        as building acquires CUDA context for current process.

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

    def _restore_state_from_checkpoint(self):
        if self._checkpoint is not None:
            external_ctx_cpt = self._pipe.RestoreFromSerializedCheckpoint(self._checkpoint)
            pipeline_data = pickle.loads(external_ctx_cpt.pipeline_data)  # nosec B301
            self._consumer_epoch_idx = self._epoch_idx = pipeline_data["epoch_idx"]
            self._consumer_iter = pipeline_data["iter"]
            if self._input_callbacks:
                for group in self._input_callbacks:
                    group.current_iter = pipeline_data["iter"]
                    group.current_sample = pipeline_data["iter"] * self._max_batch_size
            self._iterator_data = external_ctx_cpt.iterator_data
            self._is_restored_from_checkpoint = True

    def _next_op_id(self):
        i = self._next_op_id_counter
        self._next_op_id_counter += 1
        return i

    def build(self):
        """Build the pipeline (optional step).

        Instantiates the pipeline's backend objects and starts processing threads. If the pipeline
        uses multi-processing ``external_source``, the worker processes are also started.
        In most cases, there's no need to manually call build. When multi-processing is used,
        it may be necessary to call :meth:`build` or :meth:`start_py_workers` before the main
        process makes any interaction with the GPU. If needed, the :meth:`build` can used before
        running the pipeline to separate the graph building and the processing steps.

        If the pipeline requires a GPU (it contains any "cpu" or "mixed" operators or has GPU
        outputs) and no ``device_id`` was specified at construction, the current CUDA device
        (according to ``cudaGetDevice``) will be used.

        Pipeline is automatically built when it is:

            * run, either via the run APIs (:meth:`run`, :meth:`schedule_run`),
              or the framework-specific plugins,
            * the inputs are provided via :meth:`feed_input`
            * the pipeline metadata is accessed (:meth:`epoch_size`, :meth:`reader_meta`)
            * outputs are accessed - including :meth:`output_stream`
            * the graph needs to be otherwise materialized - like :meth:`save_graph_to_dot_file`.
        """
        if self._built:
            return

        if self.num_threads < 1:
            raise ValueError(
                "Pipeline created with `num_threads` < 1 can only be used " "for serialization."
            )

        self.start_py_workers()
        if not self._backend_prepared:
            self._init_pipeline_backend()
        self._setup_pipe_pool_dependency()

        self._pipe.Build(self._generate_build_args())
        self._device_id = self._pipe.device_id()
        if self._device_id == types.CPU_ONLY_DEVICE_ID:
            self._device_id = None
        self._restore_state_from_checkpoint()
        self._built = True

    def input_feed_count(self, input_name):
        self.build()
        return self._pipe.InputFeedCount(input_name)

    def _feed_input(self, name, data, layout=None, cuda_stream=None, use_copy_kernel=False):
        from nvidia.dali.external_source import _prep_data_for_feed_input

        if cuda_stream is None:
            cuda_stream = types._get_default_stream_for_array(data)
        cuda_stream_ptr = types._raw_cuda_stream_ptr(cuda_stream)

        data = _prep_data_for_feed_input(data, self._max_batch_size, layout, self._device_id)

        if isinstance(data, list):
            self._pipe.SetExternalTensorInput(name, data, cuda_stream_ptr, use_copy_kernel)
        else:
            self._pipe.SetExternalTLInput(name, data, cuda_stream_ptr, use_copy_kernel)

    def feed_input(self, data_node, data, layout=None, cuda_stream=None, use_copy_kernel=False):
        """Pass a multidimensional array or DLPack (or a list thereof) to an eligible operator.

        The operators that may be provided with data using this function are the input operators
        (i.e. everything in ``fn.inputs`` module) and the :meth:`fn.external_source`.

        In the case of the GPU input, the data must be modified on the same stream as the one
        used by ``feed_input``. See `cuda_stream` parameter for details.

        In order to avoid stalls, the data should be provided ahead of time `prefetch_queue_depth`
        times.

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

            The data to be used as the output of the operator referred to by `data_node`.

        layout : string or ``None``
            The description of the data layout (or empty string, if not specified).
            It should be a string of the length that matches the dimensionality of the data, batch
            dimension excluded. For a batch of channel-first images, this should be ``"CHW"``, for
            channel-last video it's ``"FHWC"`` and so on.
            If `data` is a DALI ``TensorList`` or a list of DALI ``Tensor`` objects and `layout`
            is ``None``, the layout is taken from `data`.
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
        self.build()
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
                (
                    _is_external_source(op) and op._callback is not None
                    for op in self._ops
                    if op.name == name
                ),
                False,
            ):
                raise RuntimeError(
                    f"Cannot use `feed_input` on the external source '{name}' with a `source`"
                    " argument specified."
                )

        self._feed_input(name, data, layout, cuda_stream, use_copy_kernel)

    def _require_exec_dynamic(self, error_message_prefix):
        if not self._exec_dynamic:
            if self._exec_separated:
                reason = "is not compatible with separated CPU/GPU queues"
            elif not self._exec_async or not self._exec_pipelined:
                reason = "requires `exec_async` and `exec_pipelined` to be enabled"
            else:
                reason = "was explicitly disabled in this pipeline"

            raise RuntimeError(error_message_prefix + " dynamic execution, which " + reason + ".")

    def outputs(self, cuda_stream=None):
        """Returns the outputs of the pipeline and releases previous buffer.

        If the pipeline is executed asynchronously, this function blocks
        until the results become available. It rises StopIteration if data set
        reached its end - usually when iter_setup cannot produce any more data.

        Parameters
        ----------
        cuda_stream : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``,
            e.g. ``cupy.cuda.Stream``, ``torch.cuda.Stream``
            The stream to which the returned `TensorLists` are bound.
            Defaults to None, which means that the outputs are synchronized with the host.
            Works only with pipelines using dynamic execution.

        Returns
        -------
            A list of `TensorList` objects for respective pipeline outputs
        """
        if cuda_stream is not None:
            self._require_exec_dynamic("Asynchronous outputs require")

        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            self._consumer_iter += 1
            if self._batches_to_consume == 0:
                raise StopIteration
            self._batches_to_consume -= 1
            return self._outputs(cuda_stream)

    def schedule_run(self):
        """Run the pipeline without returning the resulting buffers.

        If the pipeline was created with `exec_pipelined` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution. It provides better control to the users about when they
        want to run the pipeline, when they want to obtain resulting buffers
        and return them to DALI buffer pool when the results have been consumed.
        Needs to be used together with :meth:`release_outputs`
        and :meth:`share_outputs`.
        Should not be mixed with :meth:`run` in the same pipeline.

        The pipeline is built if no explicit call to ``build`` was made previously.
        """
        self.build()
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            if self._first_iter and self._exec_pipelined:
                self._prefetch()
            else:
                self._run_once()

    def output_stream(self):
        """Returns the internal CUDA stream on which the outputs are produced."""
        self.build()
        return self._pipe.GetOutputStream()

    # for the backward compatibility
    def _run(self):
        """Deprecated. Use `schedule_run` instead."""
        _show_deprecation_warning("_run", "schedule_run")
        self.schedule_run()

    def share_outputs(self, cuda_stream=None):
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

        Parameters
        ----------
        cuda_stream : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``,
            e.g. ``cupy.cuda.Stream``, ``torch.cuda.Stream``
            The stream to which the returned `TensorLists` are bound.
            Defaults to None, which means that the outputs are synchronized with the host.
            Works only with pipelines using dynamic execution.

        Returns
        -------
            A list of ``TensorList`` objects for respective pipeline outputs.
            Unless using the dynamic executor, the returned buffers are valid only until
            :meth:`release_outputs` is called.
        """
        if cuda_stream is not None:
            self._require_exec_dynamic("Asynchronous outputs require")

        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            self._consumer_iter += 1
            if self._batches_to_consume == 0:
                raise StopIteration
            self._batches_to_consume -= 1
            return self._pipe.ShareOutputs(types._raw_cuda_stream_ptr(cuda_stream))

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
        Should not be mixed with :meth:`run` in the same pipeline.

        .. note::
            When using dynamic executor (``exec_dynamic=True``), the buffers are not invalidated.
        """
        with self._check_api_type_scope(types.PipelineAPIType.SCHEDULED):
            self.build()
            ret = self._pipe.ReleaseOutputs()
            return ret

    # for the backward compatibility
    def _release_outputs(self):
        """Deprecated. Use :meth:`release_outputs` instead"""
        _show_deprecation_warning("_release_outputs", "release_outputs")
        self.release_outputs()

    def _outputs(self, cuda_stream=None):
        """Release buffers previously returned and returns  the calls.

        Calling this function is equivalent to calling release_outputs
        then calling share_outputs"""
        self.build()
        return self._pipe.Outputs(types._raw_cuda_stream_ptr(cuda_stream))

    def _are_pipeline_inputs_possible(self):
        """
        Returns True if using pipeline_inputs argument in .run() function is possible.
        """
        if not self.exec_pipelined:
            return True
        if self.exec_separated:
            return self._cpu_queue_size <= 1 and self._gpu_queue_size <= 1
        return self.prefetch_queue_depth <= 1

    def run(
        self, cuda_stream=None, /, **pipeline_inputs
    ) -> Tuple[Union[tensors.TensorListCPU, tensors.TensorListGPU], ...]:
        """
        Run the pipeline and return the result on the specified CUDA stream.

        If the pipeline was created with `exec_pipelined` option set to `True`,
        this function will also start prefetching the next iteration for
        faster execution.
        Should not be mixed with :meth:`schedule_run` in the same pipeline,
        :meth:`share_outputs` and
        :meth:`release_outputs`

        The pipeline is built if no explicit call to ``build`` was made previously.

        Parameters
        ----------
        cuda_stream : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``,
            If provided, the outputs are returned on this stream. If skipped, the results
            are host-synchronous.
            Note that with prefetch_queue_depth>1 it's possible to get host-synchronous output
            without waiting for the results of the most recent iteration.
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
            A tuple of `TensorList` objects for respective pipeline outputs
        """
        if len(pipeline_inputs) > 0 and not self._are_pipeline_inputs_possible():
            raise RuntimeError(
                f"""
                When using pipeline_inputs named arguments, either
                `prefetch_queue_depth` in Pipeline constructor shall be set to 1 (for both devices)
                or `exec_pipelined` shall be set to False.
                Received: prefetch_queue_depth={self.prefetch_queue_depth},
                exec_pipelined={self.exec_pipelined}.
                Please set the `prefetch_queue_depth` or `exec_pipelined` argument in the Pipeline
                constructor properly or provide inputs to DALI Pipeline via another mean
                (e.g. `feed_input` function or `source` argument in the `fn.external_source`
                operator.)"""
            )
        self.build()
        for inp_name, inp_value in pipeline_inputs.items():
            self.feed_input(inp_name, inp_value)
        with self._check_api_type_scope(types.PipelineAPIType.BASIC):
            self.schedule_run()
            return self.outputs(cuda_stream)

    def _prefetch(self):
        """Executes pipeline to fill executor's pipeline."""
        self.build()
        if not self._pipe:
            raise RuntimeError("The pipeline was destroyed.")
        self._schedule_py_workers()

        # We probably need some benchmarking before we remove this code path
        if not self._exec_separated:
            self._legacy_interleaved_prefetch()
            return

        # The new way: try to run the inputs and then feed them, finally call _pipe.Prefetch()
        # If this fails, we just run `_pipe.Run()` a bunch of times. This will likely blow up for
        # separated queues, which are not properly supported anyway.
        iters_fed = 0
        self._first_iter = False
        iters_fed, success = self._prefetch_inputs()
        if success:
            self._pipe.Prefetch()
        else:
            self._last_iter = True
            for _ in range(iters_fed):
                self._pipe.Run()

    # This is the old way of prefetching - the feeding and running steps are interleaved.
    # Running all callbacks at once, then feeding, then running - may affect the performance
    # of the 1st iteration.
    def _legacy_interleaved_prefetch(self):
        for _ in range(self._cpu_queue_size):
            try:
                self._first_iter = False
                self._iter_setup()
                self._batches_to_consume += 1
                if not self._exec_async and self._prefetch_queue_depth == 1:
                    self.release_outputs()
                self._pipe.Run()
            except StopIteration:
                self._last_iter = True
                break

    def _prefetch_inputs(self):
        prefetched, success = self._run_input_callbacks(True)
        self._batches_to_consume += prefetched

        if success:
            if self._exec_separated:
                prefetch_count = self._cpu_queue_size + self._gpu_queue_size
            else:
                prefetch_count = self._cpu_queue_size

            for i in range(prefetched, prefetch_count):
                try:
                    self.iter_setup()
                    prefetched = i + 1
                    self._batches_to_consume += 1
                except StopIteration:
                    success = False
                    break

        return prefetched, success

    def _run_once(self):
        """Start running the whole pipeline once without waiting for its results.

        If the pipeline was created with `exec_async` option set to `True`,
        this function will return without waiting for the execution to end."""
        self.build()
        try:
            if not self._last_iter:
                self._iter_setup()
                self._batches_to_consume += 1
            # Special case to prevent a deadlock if user didn't release the only buffer
            if not self._exec_async and self._prefetch_queue_depth == 1:
                self.release_outputs()
            if not self._last_iter:
                self._pipe.Run()
        except StopIteration:
            self._last_iter = True

    def _schedule_py_workers(self):
        if self._py_pool is None:
            return
        for i, group in enumerate(self._parallel_input_callbacks):
            group.prefetch(self._py_pool, i, self._max_batch_size, self._epoch_idx)

    def reset(self):
        """Resets pipeline iterator

        If pipeline iterator reached the end then reset its state to the beginning.
        """
        if self._last_iter:
            self._first_iter = True
            self._last_iter = False
            self._epoch_idx += 1
            if self._consumer_iter > 0:
                self._consumer_epoch_idx += 1
                self._consumer_iter = 0
            if self._input_callbacks:
                for group in self._input_callbacks:
                    group.reset_indices()
                for i, group in enumerate(self._parallel_input_callbacks):
                    # iterators are not reset or their prefetch results discarded
                    # unless they have caused an exception
                    if not self._py_pool.is_iterable_group(group):
                        self._py_pool.reset_context(i)

    def empty(self):
        """If there is any work scheduled in the pipeline but not yet consumed"""
        return self._batches_to_consume == 0

    def serialize(self, define_graph=None, filename=None):
        """Serialize the pipeline definition to a Protobuf string.

        .. note::
            This function doesn't serialize the pipeline's internal state. Use
            `checkpointing <advanced_topics_checkpointing.html>`_ to achieve that.

        Additionally, you can pass a file name, so that serialized pipeline will be written there.
        The file contents will be overwritten.

        Parameters
        ----------
        define_graph : callable, optional, default = None
                Deprecated.
                If specified, this function will be used instead of member :meth:`define_graph`.
                This parameter must not be set, if the pipeline outputs are specified with
                :meth:`set_outputs` or the pipeline was constructed with a function decorated with
                :meth:`pipeline_def`.
        filename : str, optional, default = None
                The file that the serialized pipeline definition will be written to.
        """
        if define_graph is not None and not callable(define_graph):
            raise TypeError(
                "Provided `define_graph` argument is not callable."
                + (
                    " Didn't you want to write `.serialize(filename=...)`?"
                    if isinstance(define_graph, str)
                    else ""
                )
            )
        if not self._py_graph_built:
            self._build_graph(define_graph)
        if not self._backend_prepared:
            self._init_pipeline_backend()
            self._pipe.SetOutputDescs(self._generate_build_args())
        ret = self._pipe.SerializeToProtobuf()
        if filename is not None:
            with open(filename, "wb") as pipeline_file:
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

        Note, that `serialized_pipeline` and `filename` parameters are mutually exclusive

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
                "Precisely one of them should be defined."
            )
        pipeline = cls()
        if filename is not None:
            with open(filename, "rb") as pipeline_file:
                serialized_pipeline = pipeline_file.read()

        prefetch_queue_depth = kw.get("prefetch_queue_depth", None)
        exec_separated = False
        if isinstance(prefetch_queue_depth, int):
            prefetch_queue_depths = (prefetch_queue_depth, prefetch_queue_depth)
        elif isinstance(prefetch_queue_depth, dict):
            if "cpu" not in prefetch_queue_depth and "gpu" not in prefetch_queue_depth:
                raise ValueError("prefetch_queue_depth must contain either 'cpu' or 'gpu' key")
            exec_separated = True
            prefetch_queue_depths = (
                prefetch_queue_depth.get("cpu", prefetch_queue_depth["gpu"]),
                prefetch_queue_depth.get("gpu", prefetch_queue_depth["cpu"]),
            )
        else:
            prefetch_queue_depths = None

        exec_pipelined = kw.get("exec_pipelined", None)
        exec_async = kw.get("exec_async", None)
        exec_dynamic = kw.get("exec_dynamic", None)
        stream_policy = kw.get("stream_policy", None)
        concurrency = kw.get("concurrency", None)
        executor_type = b._MakeExecutorType(
            exec_pipelined or False, exec_async or False, exec_separated, exec_dynamic or False
        )
        executor_flags = b._ExecutorFlags.NoFlags
        if kw.get("set_affinity", False):
            executor_flags |= b._ExecutorFlags.SetAffinity

        if stream_policy is not None:
            executor_flags &= ~b._ExecutorFlags.StreamPolicyMask
            executor_flags |= stream_policy.value
        if concurrency is not None:
            executor_flags &= ~b._ExecutorFlags.ConcurrencyMask
            executor_flags |= concurrency.value

        seed = kw.get("seed", None)
        if seed is not None and seed < 0:
            warnings.warn(
                "The use of negative seed to use automatic seed asignment is deprecated. "
                "Use `None` instead."
            )
            seed = None

        params = b._PipelineParams(
            max_batch_size=kw.get("batch_size", None),
            num_threads=kw.get("num_threads", None),
            device_id=kw.get("device_id", None),
            seed=seed,
            executor_type=executor_type,
            executor_flags=executor_flags,
            prefetch_queue_depths=prefetch_queue_depths,
            enable_checkpointing=kw.get("enable_checkpointing", None),
            enable_memory_stats=kw.get("enable_memory_stats", None),
            bytes_per_sample_hint=kw.get("bytes_per_sample", None),
        )
        pipeline._pipe = b.Pipeline(serialized_pipeline, params)
        if pipeline._pipe.requires_gpu():
            b.check_cuda_runtime()
        pipeline._set_params(pipeline._pipe.params())

        pipeline._backend_prepared = True
        pipeline._pipe.Build()
        pipeline._device_id = pipeline._pipe.device_id()
        if pipeline._device_id == types.CPU_ONLY_DEVICE_ID:
            pipeline._device_id = None
        pipeline._restore_state_from_checkpoint()
        pipeline._built = True
        pipeline._deserialized = True
        return pipeline

    def deserialize_and_build(self, serialized_pipeline):
        """Deserialize and build the pipeline given in serialized form.

        Parameters
        ----------
        serialized_pipeline : str
                              Serialized pipeline.
        """
        self._pipe = b.Pipeline(serialized_pipeline, self._get_params())
        self._set_params(self._pipe.params())
        self._backend_prepared = True
        self._pipe.Build()
        self._device_id = self._pipe.device_id()
        if self._device_id == types.CPU_ONLY_DEVICE_ID:
            self._device_id = None

        self._restore_state_from_checkpoint()
        self._built = True
        self._deserialized = True

    def save_graph_to_dot_file(
        self, filename, *, show_tensors=False, show_ids=None, use_colors=False
    ):
        """Saves the pipeline graph to a file.

        Parameters
        ----------
        filename : str
                   Name of the file to which the graph is written.
        show_tensors : bool
                   Show the Tensor nodes in the graph (by default only Operator nodes are shown)
        show_ids : bool, deprecated
                   This flag is obsolete and has no effect
        use_colors : bool
                   Whether use color to distinguish stages
        """
        self.build()
        if show_ids is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                msg = (
                    'The argument "show_ids" is deprecated because it no longer has any effect.\n'
                    "It will be removed from future releases."
                )
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

        self._pipe.SaveGraphToDotFile(filename, show_tensors, use_colors)

    def _get_checkpoint(self, iterator_data=""):
        """
        Returns the pipeline's state as a serialized Protobuf string.
        Also, allows to pass additional data to be saved in the checkpoint.
        """

        external_ctx_cpt = b.ExternalContextCheckpoint()
        external_ctx_cpt.pipeline_data = pickle.dumps(
            {"iter": self._consumer_iter, "epoch_idx": self._epoch_idx}
        )
        external_ctx_cpt.iterator_data = iterator_data
        return self._pipe.GetSerializedCheckpoint(external_ctx_cpt)

    def checkpoint(self, filename=None):
        """Returns the pipeline's state as a serialized Protobuf string.

        Additionally, if `filename` is specified, the serialized checkpoint will be
        written to the specified file. The file contents will be overwritten.

        The same pipeline can be later rebuilt with the saved checkpoint passed as a `checkpoint`
        parameter to resume execution from the saved iteration.

        More details can be found in
        `this documentation section <advanced_topics_checkpointing.html>`_.

        Parameters
        ----------
        filename : str
                The file that the serialized pipeline will be written to.
        """
        self.build()
        cpt = self._get_checkpoint()
        if filename is not None:
            with open(filename, "wb") as checkpoint_file:
                checkpoint_file.write(cpt)
        return cpt

    def set_outputs(self, *output_data_nodes):
        """Set the outputs of the pipeline.

        Use of this function is an alternative to overriding `define_graph` in a derived class.

        Args
        ----
        *output_data_nodes : unpacked list of :class:`DataNode` objects
            The outputs of the pipeline
        """
        self._graph_out = output_data_nodes

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.

        It returns a list of outputs created by calling DALI Operators."""
        raise NotImplementedError

    def _iter_setup(self):
        self.build()
        iters, success = self._run_input_callbacks()
        if not success:
            raise StopIteration
        if iters == 0:
            self.iter_setup()

    def _run_input_callbacks(self, is_prefetch=False):
        if self._input_callbacks is None:
            return 0, True

        done = False
        stop_iter = False
        iter = 0
        while not done and not stop_iter:
            done = True
            batches = []  # data from external source callbacks is gathered here
            for i, group in enumerate(self._parallel_input_callbacks):
                try:
                    count = group.feed_count(self) if is_prefetch else 1
                    if iter < count:
                        batches.append(
                            group.schedule_and_receive(
                                self, self._py_pool, i, self._max_batch_size, self._epoch_idx
                            )
                        )
                        if iter + 1 < count:
                            done = False
                except StopIteration:
                    stop_iter = True
            for group in self._seq_input_callbacks:
                try:
                    count = group.feed_count(self) if is_prefetch else 1
                    if iter < count:
                        batches.append(group.get_batch(self, self._max_batch_size, self._epoch_idx))
                        if iter + 1 < count:
                            done = False
                except StopIteration:
                    stop_iter = True

            if stop_iter:
                return iter, False

            try:
                self.iter_setup()
            except StopIteration:
                return iter, False

            # we only fill external source queues when we know that all callbacks succeeded
            for batch in batches:
                batch.feed()

            iter += 1
        return iter, True

    def iter_setup(self):
        """A deprecated method of providing the pipeline with external inputs.

        This function can be overridden by a user-defined
        pipeline to perform any needed setup for each iteration.
        For example, one can use this function to feed the input
        data from NumPy arrays.

        This method is deprecated and its use is discouraged. Newer execution models may be
        incompatible with this method of providing data to the pipeline. Use `source` argument
        in ``external_source`` instead, where possible.
        """
        pass

    def _generate_build_args(self):
        num_outputs = len(self._names_and_devices)
        dtypes = (
            [self._output_dtype] * num_outputs
            if type(self._output_dtype) is not list
            else self._output_dtype
        )
        ndims = (
            [self._output_ndim] * num_outputs
            if type(self._output_ndim) is not list
            else self._output_ndim
        )
        layouts = (
            [self._output_layout] * num_outputs
            if type(self._output_layout) is not list
            else self._output_layout
        )
        if not (len(dtypes) == len(ndims) == num_outputs):
            raise RuntimeError(
                f"Lengths of provided output descriptions do not match. \n"
                f"Expected num_outputs={num_outputs}."
                f"\nReceived:\noutput_dtype={dtypes}\noutput_ndim={ndims}"
            )

        return [
            (
                name,
                dev,
                types.NO_TYPE if dtype is None else dtype,
                -1 if ndim is None else ndim,
                layout if layout is not None else "",
            )
            for (name, dev), dtype, ndim, layout in zip(
                self._names_and_devices, dtypes, ndims, layouts
            )
        ]

    def _stub(self):
        """Produce a stub by shallow-copying the pipeline, removing the backend and forbidding
        operations that require the backend.

        Stub pipelines are necessary in contexts where passing the actual pipeline would cause
        circular reference - notably, PythonFunction operator.
        """
        stub = copy.copy(self)
        stub._pipe = None

        def short_circuit(self, *args, **kwargs):
            raise RuntimeError("This method is forbidden in current context")

        stub.start_py_workers = short_circuit
        stub.build = short_circuit
        stub.run = short_circuit
        stub.schedule_run = short_circuit
        stub.outputs = short_circuit
        stub.share_outputs = short_circuit
        stub.release_outputs = short_circuit
        stub.add_sink = short_circuit
        stub.checkpoint = short_circuit
        stub.set_outputs = short_circuit
        stub.executor_statistics = short_circuit
        stub.external_source_shm_statistics = short_circuit
        return stub


def _shutdown_pipelines():
    for weak in list(Pipeline._pipes):
        p = weak()
        if p is None:
            Pipeline._pipes.discard(weak)
            continue
        p._shutdown()
    assert len(Pipeline._pipes) == 0


# Shut down the pipelines, so that nothing is running when the interpreter is torn down
atexit.register(_shutdown_pipelines)


def _discriminate_args(func, **func_kwargs):
    """Split args on those applicable to Pipeline constructor and the decorated function."""
    func_argspec = inspect.getfullargspec(func)
    ctor_argspec = inspect.getfullargspec(Pipeline.__init__)

    if "debug" not in func_argspec.args and "debug" not in func_argspec.kwonlyargs:
        func_kwargs.pop("debug", False)

    if (
        "enable_conditionals" not in func_argspec.args
        and "enable_conditionals" not in func_argspec.kwonlyargs
    ):
        func_kwargs.pop("enable_conditionals", False)

    ctor_args = {}
    fn_args = {}

    if func_argspec.varkw is not None:
        raise TypeError(
            f"Using variadic keyword argument `**{func_argspec.varkw}` in a  "
            f"graph-defining function is not allowed."
        )

    for farg in func_kwargs.items():
        is_ctor_arg = farg[0] in ctor_argspec.args or farg[0] in ctor_argspec.kwonlyargs
        is_fn_arg = farg[0] in func_argspec.args or farg[0] in func_argspec.kwonlyargs
        if is_fn_arg:
            fn_args[farg[0]] = farg[1]
            if is_ctor_arg:
                print(
                    f"Warning: the argument `{farg[0]}` shadows a Pipeline constructor "
                    "argument of the same name."
                )
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
    """Transform the pipeline definition function if the conditionals are enabled"""
    if conditionals_on:
        return _conditionals._autograph.convert(recursive=True, user_requested=True)(func)
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
            po = (pipe_outputs,)
        pipe.set_outputs(*po)


# Based on: https://mypy.readthedocs.io/en/stable/generics.html#decorator-factories
# Tuple[DataNode, ...] is considered a variable length tuple of uniform DataNode contents
# Bare decorator usage
@overload
def pipeline_def(
    __func: Callable[..., Union[DataNode, Tuple[DataNode, ...]]],
) -> Callable[..., Pipeline]: ...


# Decorator with arguments
@overload
def pipeline_def(
    *,
    enable_conditionals: bool = False,
    batch_size: int = -1,
    num_threads: int = -1,
    device_id: int = -1,
    seed: int = -1,
    exec_pipelined: bool = True,
    prefetch_queue_depth: Union[int, Tuple[int, int]] = 2,
    exec_async: bool = True,
    bytes_per_sample: int = 0,
    set_affinity: bool = False,
    enable_memory_stats: bool = False,
    enable_checkpointing: bool = False,
    checkpoint: Optional[Any] = None,
    py_num_workers: int = 1,
    py_start_method: str = "fork",
    py_callback_pickler: Optional[Any] = None,
    output_dtype: Union[types.DALIDataType, Tuple[types.DALIDataType, ...], None] = None,
    output_ndim: Union[int, Tuple[int, ...], None] = None,
) -> Callable[[Callable[..., Union[DataNode, Tuple[DataNode, ...]]]], Callable[..., Pipeline]]: ...


# Implementation
def pipeline_def(
    fn: Optional[Callable[..., Any]] = None, *, enable_conditionals: bool = False, **pipeline_kwargs
):
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
        # pipe.run()  # the pipeline is not configured properly yet

    A pipeline requires additional parameters such as batch size, number of worker threads,
    GPU device id and so on (see :meth:`nvidia.dali.Pipeline()` for a
    complete list of pipeline parameters).
    These parameters can be supplied as additional keyword arguments,
    passed to the decorated function::

        pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)

    The pipeline is properly configured, we can run it now. The outputs from the original function
    became the outputs of the Pipeline::

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
        if _conditionals._autograph.is_autograph_artifact(func):
            raise ValueError("Pipeline definition cannot be marked with @do_not_convert.")

        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            conditionals_on = kwargs.get("enable_conditionals", enable_conditionals)

            pipe_func = _preprocess_pipe_func(func, conditionals_on)
            # TODO(klecki): Rewrite _discriminate_args used by _regroup_args in the means of the
            # inspect.signature, so it obeys the signature produced by the wrapper.
            # The getfullargspec ignores wrappers, so we need to use func here for argument
            # redistribution, as _preprocess_pipe_func returns a wrapper in conditional mode.
            # After this we can pass pipe_func below.
            pipeline_args, fn_kwargs = _regroup_args(func, pipeline_kwargs, kwargs)
            pipe = Pipeline(**pipeline_args)
            _preprocess_pipe_object(pipe, conditionals_on, args, fn_kwargs)

            _generate_graph(pipe, pipe_func, args, fn_kwargs)
            return pipe

        # Add `is_pipeline_def` attribute to the function marked as `@pipeline_def`
        create_pipeline._is_pipeline_def = True
        return create_pipeline

    return actual_decorator(fn) if fn else actual_decorator


# Callable preserving a signature
_F = TypeVar("_F", bound=Callable[..., Any])


def do_not_convert(func: _F = None) -> _F:
    """Decorator that suppresses the conversion of a function by AutoGraph.

    In conditional mode, DALI uses a fork of
    `TensorFlow's AutoGraph <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md>`_
    to transform the code, enabling us to rewrite and detect the ``if`` statements, so they can be
    used in processing the DALI pipeline.

    The AutoGraph conversion is applied to any top-level function or method called within the
    pipeline definition (as well as the pipeline definition itself).
    When a function is converted, all functions defined within its syntactical scope are also
    converted. The rewriting, among other effects, makes these functions non-serializable.

    To stop a function from being converted, its top-level encompassing function must be marked
    with this decorator. This may sometimes require refactoring the function to outer scope.

    Parallel mode of :meth:`external source <nvidia.dali.fn.external_source>` (``parallel=True``),
    requires that its ``source`` parameter is serializable. To prevent the rewriting of the
    ``source``, the functions that are used to create the ``source``,
    should be decorated with :meth:`@do_not_convert <nvidia.dali.pipeline.do_not_convert>`.

    .. note::
       Only functions that do not process :class:`DataNode` (so do not use DALI operators)
       should be marked with this decorator.

    For example::

        from nvidia.dali import pipeline_def, fn

        @pipeline_def(enable_conditionals=True)
        def pipe():

            def source_factory(size):
                def source_fun(sample_info):
                    return np.full(size, sample_info.iter_idx)
                return source_fun

            source = source_factory(size=(2, 1))
            return fn.external_source(source=source, parallel=True, batch=False)

    Should be converted into::

        from nvidia.dali import pipeline_def, fn
        from nvidia.dali.pipeline import do_not_convert

        @do_not_convert
        def source_factory(size):
            def source_fun(sample_info):
                return np.full(size, sample_info.iter_idx)
            return source_fun

        @pipeline_def(enable_conditionals=True)
        def pipe():
            source = source_factory(size=(2, 1))
            return fn.external_source(source=source, parallel=True, batch=False)

    The ``source_factory`` must be factored out, otherwise it would be converted as a part of
    pipeline definition. As we are interested in preventing the AutoGraph conversion of
    ``source_fun`` we need to decorate its top-level encompassing function.

    .. note::
       If a function is declared outside of the pipeline definition, and is passed as a parameter,
       but not directly invoked within the pipeline definition, it will not be converted.
       In such case, a callback passed to
       :meth:`external source <nvidia.dali.fn.external_source>` operator,
       :meth:`python function <nvidia.dali.fn.python_function>` operator family or
       :meth:`Numba function <nvidia.dali.plugin.numba.fn.experimental.numba_function>` operator
       is not considered as being directly invoked in pipeline definition. Such callback is
       executed when the pipeline is run, so after the pipeline is defined and built.

    For example::

        from nvidia.dali import pipeline_def, fn

        def source_fun(sample_info):
            return np.full((2, 2), sample_info.iter_idx)

        @pipeline_def(enable_conditionals=True)
        def pipe():
            return fn.external_source(source=source_fun, batch=False)

    The ``source_fun`` won't be converted, as it is defined outside of pipeline definition and
    it is only passed via name to external source.
    """  # noqa(E501)

    if func is None:
        return do_not_convert

    if getattr(func, "_is_pipeline_def", False):
        raise ValueError("Pipeline definition cannot be marked with @do_not_convert.")

    # Marking a function as autograph_artifact will prevent it from being converted without
    # adding any intermediate functions or adjusting the code. This is more lightweight solution
    # that should keep numba happy.
    return _conditionals._autograph.autograph_artifact(func)


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

    visited = set()
    ops = []

    # Depth-first search returns the graph topologically sorted.
    # We go over each operator's inputs before adding it to the list.

    def visit_op(op):
        if id(op) in visited:
            return
        visited.add(id(op))
        op.check_args()
        # visit conttributing inputs
        for edge in get_op_input_edges(op):
            visit_op(get_source_op(edge))
        # add the operator to the list of contributing ops
        ops.append(op)

    for edge in output_nodes:
        visit_op(get_source_op(edge))

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

    pipeline_debug = pipeline_kwargs.pop("debug", False)

    def actual_decorator(func):
        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            debug_mode_on = kwargs.get("debug", pipeline_debug)
            conditionals_on = kwargs.get("enable_conditionals", enable_conditionals)

            pipe_func = _preprocess_pipe_func(func, conditionals_on)
            # TODO(klecki): Use pipe_func here after similar todo is resolved in regular decorator.
            pipeline_args, fn_kwargs = _regroup_args(func, pipeline_kwargs, kwargs)
            if debug_mode_on:
                pipe = _PipelineDebug(
                    functools.partial(pipe_func, *args, **fn_kwargs), **pipeline_args
                )
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
    experimental_module = internal.get_submodule(current_module, "experimental")
    _pipeline_def_experimental.__module__ = experimental_module
    setattr(experimental_module, "pipeline_def", _pipeline_def_experimental)


_insert_experimental_pipeline_def()
