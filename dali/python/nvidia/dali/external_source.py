# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# custom wrappers around ops
from nvidia.dali import backend as _b
from nvidia.dali import tensors as _tensors
from nvidia.dali import types as _types
from nvidia.dali._multiproc.messages import TaskArgs as _TaskArgs, SampleRange as _SampleRange
import nvidia.dali.types
from nvidia.dali._utils.external_source_impl import \
        get_callback_from_source as _get_callback_from_source, \
        accepted_arg_count as _accepted_arg_count, \
        SourceKind as _SourceKind


def _get_batch_shape(data):
    if isinstance(data, (list, tuple, _tensors.TensorListCPU, _tensors.TensorListGPU)):
        if len(data) == 0:
            return [], True
        if callable(data[0].shape):
            return [x.shape() for x in data], False
        else:
            return [x.shape for x in data], False
    else:
        shape = data.shape
        if callable(shape):
            shape = data.shape()
        return [shape[1:]] * shape[0], True


def _check_data_batch(data, batch_size, layout):
    shape, uniform = _get_batch_shape(data)
    if len(shape) > batch_size:
        raise RuntimeError(f"The external source callback returned an unexpected batch "
                           f"size. Expected batch_size <= {batch_size}, actual: {len(shape)}")

    if len(shape) > 0:
        dim = len(shape[0])
        if not uniform:
            for ts in shape:
                if len(ts) != dim:
                    raise RuntimeError(
                        "All tensors in a batch must have the same number of dimensions")
        if layout is not None and layout != "" and dim != len(layout):
            raise RuntimeError(f"The layout '{layout}' cannot describe {dim}-dimensional data")


def _prep_data_for_feed_input(data, batch_size, layout, device_id=None):

    def to_numpy(x):
        if _types._is_mxnet_array(x):
            return x.asnumpy()
        elif _types._is_torch_tensor(x):
            return x.numpy()
        else:
            return x

    # __cuda_array_interface__ doesn't provide any way to pass the information about the device
    # where the memory is located. It is assumed that the current device is the one that
    # the memory belongs to, unless the user sets the device explicitly
    # creating TensorGPU/TensorListGPU
    if isinstance(data, (_tensors.TensorListCPU, _tensors.TensorListGPU)):
        if layout is not None:
            _check_data_batch(data, batch_size, layout)
            data = type(data)(data, layout)
    elif isinstance(data, list):
        inputs = []
        checked = False
        for datum in data:
            (is_dlpack, is_gpu_data) = _b.CheckDLPackCapsule(datum)
            if not is_dlpack and not checked:
                _check_data_batch(data, batch_size, layout)
                checked = True
            if isinstance(datum, (_tensors.TensorCPU, _tensors.TensorGPU)):
                inp = type(datum)(datum, layout=layout) if layout is not None else datum
            elif is_dlpack:
                if is_gpu_data:
                    inp = _tensors.TensorGPU(datum, layout or "")
                else:
                    inp = _tensors.TensorCPU(datum, layout or "")
            elif hasattr(datum, "__cuda_array_interface__"):
                array_device_id = _types._get_device_id_for_array(datum)
                if array_device_id is None:
                    array_device_id = device_id
                inp = _tensors.TensorGPU(datum, layout or "", array_device_id)
            else:
                datum = to_numpy(datum)
                inp = _tensors.TensorCPU(datum, layout or "")
            inputs.append(inp)
        assert all(isinstance(inp, type(inputs[0])) for inp in inputs), \
            "Mixed input types are not support, all need to reside on the CPU or GPU"
        data = inputs
    else:
        (is_dlpack, is_gpu_data) = _b.CheckDLPackCapsule(data)
        if not is_dlpack:
            _check_data_batch(data, batch_size, layout)
        if hasattr(data, "__cuda_array_interface__"):
            array_device_id = _types._get_device_id_for_array(data)
            if array_device_id is None:
                array_device_id = device_id
            data = _tensors.TensorListGPU(data, layout or "", array_device_id)
        elif is_dlpack:
            if is_gpu_data:
                data = _tensors.TensorListGPU(data, layout or "")
            else:
                data = _tensors.TensorListCPU(data, layout or "")
        else:
            data = to_numpy(data)
            data = _tensors.TensorListCPU(data, layout or "")
    return data


class _ExternalDataBatch:

    def __init__(self, group, pipeline, data, batch_size):
        self._group = group
        self._pipepline = pipeline
        self._data = data
        self._batch_size = batch_size

    def feed(self):
        self._group.feed(self._pipepline, self._data, self._batch_size)


class _ExternalSourceGroup(object):

    def __init__(self, callback, source_desc, is_multioutput, instances=[], *,
                 cuda_stream=None, use_copy_kernel=None, batch=True, parallel=False,
                 prefetch_queue_depth=None, batch_info=None):
        self.instances = list(instances)  # we need a copy!
        self.utilized_instances = self.instances
        self.is_multioutput = is_multioutput
        self.callback = callback
        self.source_desc = source_desc
        self._cuda_stream = cuda_stream
        self.use_copy_kernel = use_copy_kernel
        self.batch = batch
        self.batch_info = batch_info
        # Index of a batch within the epoch that will be returned from
        # get_batch or schedule_and_receive call. Contrary to Pipeline's
        # `epoch_idx` it is tracked separately by ExternalSourceGroup due to
        # prefetching of batches in parallel mode.
        self.current_iter = 0
        self.current_sample = 0
        self.parallel = parallel
        self.prefetch_queue_depth = prefetch_queue_depth
        if callback is not None:
            arg_count = _accepted_arg_count(callback)
            if arg_count not in [0, 1]:
                raise TypeError("External source callback must be a callable with 0 or 1 argument")
            self.accepts_arg = arg_count > 0

    def append(self, instance):
        self.instances.append(instance)
        self.utilized_instances = self.instances

    def disable_pruned_instances(self, pruned_mask):
        if len(pruned_mask) != len(self.instances):
            raise RuntimeError(
                f"Mask of the pruned outputs of the external source must have the length matching "
                f"the number of outputs of the external source. The external source node has "
                f"{len(self.instances)} outputs, but received mask of length {len(pruned_mask)}.")
        self.utilized_instances = [
            instance for instance, is_pruned
            in zip(self.instances, pruned_mask) if not is_pruned
        ]

    def callback_args(self, idx_in_batch, epoch_idx, batch_size=0, lead=0):
        """Generate information to be passed to ES callback.

        Args:
            idx_in_batch: Index in batch for per-sample mode, None indicates batch mode where we
            pass only the iteration number.
            lead: how many batches ahead is this job wrt actual iteration
        """
        if not self.accepts_arg:
            return ()
        if idx_in_batch is not None:
            arg = nvidia.dali.types.SampleInfo(
                self.current_sample + idx_in_batch + batch_size * lead, idx_in_batch,
                self.current_iter + lead, epoch_idx)
        elif self.batch_info:
            arg = nvidia.dali.types.BatchInfo(self.current_iter + lead, epoch_idx)
        else:
            arg = self.current_iter + lead
        return (arg, )

    def reset_indices(self):
        self.current_iter = 0
        self.current_sample = 0

    def prefetch(self, pool, context_i, batch_size, epoch_idx):
        # NOTE We can't schedule more than what's on top of pipeline's prefetch queue, as the
        # entires in the pipeline are zero-copy and cannot be overwritten.
        context = pool.contexts[context_i]
        while (context.scheduled_ahead < self.prefetch_queue_depth
               and self.schedule_batch(pool, context_i, context.scheduled_ahead,
                                       batch_size, epoch_idx)):
            pass

    def schedule_batch(self, pool, context_i, lead, batch_size, epoch_idx):
        """Schedule computing new batch from source callback by the parallel pool."""
        if self.batch:
            return pool.schedule_batch(
                context_i, _TaskArgs.make_batch(self.callback_args(None, epoch_idx, lead=lead)))
        else:
            sample_range_start = self.current_sample + batch_size * lead
            sample_range_end = sample_range_start + batch_size
            iteration = self.current_iter + lead
            sample_range = _SampleRange(sample_range_start, sample_range_end, iteration, epoch_idx)
            work_batch = _TaskArgs.make_sample(sample_range)
            return pool.schedule_batch(context_i, work_batch)

    def schedule_and_receive(self, pipeline, pool, context_i, batch_size, epoch_idx):
        """Obtain the computed results of calling source callback in parallel pool and feed
        the results to the ExternalSource nodes in `pipeline`.
        Schedule the execution of the source callback in the pool to compute next batch.
        Used by the parallel ExternalSource variant.

        Args:
            context_i (int): Index of the callback (in the list of parallel groups)"""
        try:
            callback_out = pool.receive_batch(context_i)
            self.current_sample += batch_size
            self.current_iter += 1
            self.prefetch(pool, context_i, batch_size, epoch_idx)
            return _ExternalDataBatch(self, pipeline, callback_out, batch_size)
        except StopIteration:
            self.reset_indices()
            pool.reset_context(context_i)
            raise

    def get_batch(self, pipeline, batch_size, epoch_idx):
        """Call the source callback and feed the results to the ExternalSource nodes in `pipeline`.
        Used for the sequential ExternalSource variant."""
        try:
            if self.batch:
                callback_out = self.callback(*self.callback_args(None, epoch_idx))
            else:
                callback_out = [
                    self.callback(*self.callback_args(i, epoch_idx)) for i in range(batch_size)
                ]
            self.current_sample += batch_size
            self.current_iter += 1
        except StopIteration:
            self.reset_indices()
            raise
        return _ExternalDataBatch(self, pipeline, callback_out, batch_size)

    def feed(self, pipeline, callback_out, batch_size):
        """Feed the `callback_out` data obtained from source to the ExternalSource nodes
        in the `pipeline`"""
        if self.is_multioutput:
            for op in self.utilized_instances:
                if self.batch:
                    data = callback_out[op._output_index]
                else:
                    # extract a single output
                    data = [callback_out[i][op._output_index] for i in range(batch_size)]
                pipeline._feed_input(op._name, data, op._layout, self._cuda_stream,
                                     self.use_copy_kernel)
        else:
            data = callback_out
            op = self.utilized_instances[0]
            pipeline._feed_input(
                op._name, data, op._layout, self._cuda_stream, self.use_copy_kernel)


class ExternalSource():
    """ExternalSource is a special operator that can provide data to a DALI pipeline
from Python by several methods.

The simplest and preferred way is to specify a ``source``, which can be a callable or iterable.

.. note::
    :meth:`nvidia.dali.fn.external_source` operator is partially compatible with TensorFlow
    integration via :meth:`nvidia.dali.plugin.tf.experimental.DALIDatasetWithInputs`.
    Please refer to its documentation for details.

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
"""

    _args_doc = """
Args
----

`source` : callable or iterable
    The source of the data.

    The source is polled for data (via a call ``source()`` or ``next(source)``)
    when the pipeline needs input for the next iteration. Depending on the value of ``num_outputs``,
    the source can supply one or more data items. The data item can be a whole batch (default) or
    a single batch entry (when ``batch==False``). If ``num_outputs`` is not set, the ``source``
    is expected to return one item (a batch or a sample). If this value is specified (even if its
    value is 1), the data is expected to a be tuple, or list, where each element corresponds to
    respective return value of the external_source.

    The data samples must be in one of the compatible array types:

        * NumPy ndarray (CPU)
        * MXNet ndarray (CPU)
        * PyTorch tensor (CPU or GPU)
        * CuPy array (GPU)
        * objects implementing ``__cuda_array_interface__``
        * DALI `Tensor` object

    Batch sources must produce entire batches of data. This can be achieved either by adding a new
    outermost dimension to an array or by returning a list of arrays (in which case they can be of
    different size, but must have the same rank and element type). A batch source can also
    produce a DALI `TensorList` object, which can be an output of another DALI pipeline.

    A per-batch source may accept one positional argument. If it does, it is the index of current
    iteration within epoch and consecutive calls will be ``source(0)``, ``source(1)``, and so on.
    If `batch_info` is set to True, instance of :class:`nvidia.dali.types.BatchInfo` will be
    passed to the source, instead of a plain index.

    A per-sample source may accept one positional argument of type
    :class:`nvidia.dali.types.SampleInfo`, which contains index of the sample in current epoch and
    in the batch, as well as current iteration number.

    If the source is a generator function, the function is invoked and treated as an iterable.
    However, unlike a generator, the function can be used with ``cycle``. In this case, the function
    will be called again when the generator reaches the end of iteration.

    For GPU inputs, it is a user's responsibility to modify the provided GPU memory content
    only in the provided stream. DALI schedules a copy on this stream, and all work is properly
    queued. If no stream is provided, DALI will use a default, with a best-effort approach at
    correctness. See the ``cuda_stream`` argument documentation for more information.

`num_outputs` : int, optional
    If specified, denotes the number of TensorLists that are produced by the source function.

    If set, the operator returns a list of ``DataNode`` objects, otherwise a single ``DataNode``
    object is returned.

Keyword Args
------------

`cycle`: string or bool, optional
    Specifies if and how to cycle through the source.
    It can be one of the following values:

        * ``"no"``, ``False`` or ``None`` - don't cycle; ``StopIteration`` is raised when
          end of data is reached; this is the default behavior
        * ``"quiet"`` or ``True`` - the data is repeated indefinitely,
        * ``"raise"`` - when the end of data is reached, ``StopIteration`` is raised, but
          the iteration is restarted on subsequent call.

    This flag requires that the ``source`` is a collection, for example, an iterable object where
    ``iter(source)`` returns a fresh iterator on each call, or a generator function.
    In the latter case, the generator function is called again when more data than was
    yielded by the function is requested.

    Specifying ``"raise"`` can be used with DALI iterators to create a notion of epoch.

`name` : str, optional
    The name of the data node.

    Used when feeding the data in ``iter_setup`` and can be omitted if
    the data is provided by ``source``.

`layout` : :ref:`layout str<layout_str_doc>` or list/tuple thereof, optional
    If provided, sets the layout of the data.

    When ``num_outputs > 1``, the layout can be a list that contains a distinct layout for each
    output. If the list has fewer than ``num_outputs`` elements, only the first
    outputs have the layout set, the rest of the outputs don't have a layout set.

`dtype` : `nvidia.dali.types.DALIDataType` or list/tuple thereof, optional
    Input data type.

    When ``num_outputs > 1``, the ``dtype`` can be a list that contains a distinct value for each
    output.

    The operator will validate that the fetched data is of the provided type.
    If the argument is omitted or :const:`DALIDataType.NO_TYPE` is passed, the operator will infer
    the type from the provided data.

    This argument will be required starting from DALI 2.0.

`ndim` : int or list/tuple thereof, optional
    Number of dimensions in the input data.

    When ``num_outputs > 1``, the ``ndim`` can be a list that contains a distinct value for each
    output.

    The dimensionality of the data provided to the operator will be verified against this value.
    Number of dimensions can be also inferred from the ``layout`` argument if provided.

    If the ``layout`` argument is provided, the ``ndim`` must match
    the number of dimensions in the layout.

    Specifying the input dimensionality will be required starting from DALI 2.0

`cuda_stream` : optional, ``cudaStream_t`` or an object convertible to ``cudaStream_t``,
    such as ``cupy.cuda.Stream`` or ``torch.cuda.Stream``
    The CUDA stream is used to copy data to the GPU or from a GPU source.

    If this parameter is not set, a best-effort will be taken to maintain correctness. That is,
    if the data is provided as a tensor/array from a recognized library such as CuPy or PyTorch,
    the library's current stream is used. Although this approach works in typical scenarios,
    with advanced use cases, and code that uses unsupported libraries, you might need to
    explicitly supply the stream handle.

    This argument has two special values:

      * 0 - Use the default CUDA stream
      * 1 - Use DALI's internal stream

    If internal stream is used, the call to ``feed_input`` will block until the copy to internal
    buffer is complete, since there's no way to synchronize with this stream to prevent
    overwriting the array with new data in another stream.

`use_copy_kernel` : bool, optional
    If set to True, DALI will use a CUDA kernel to feed the data
    instead of cudaMemcpyAsync (default).

    .. note::
        This is applicable only when copying data to and from GPU memory.

`blocking` : bool, optional
    Determines whether the external source should wait until data is available or just fail
    when the data is not available.

`no_copy` : bool, optional
    Determines whether DALI should copy the buffer when feed_input is called.

    If set to True, DALI passes the user memory directly to the pipeline, instead of copying it.
    It is the user responsibility to keep the buffer alive and unmodified until it is
    consumed by the pipeline.

    The buffer can be modified or freed again after the output of the relevant iterations
    has been consumed. Effectively, it happens after Pipeline's ``prefetch_queue_depth`` or
    ``cpu_queue_depth * gpu_queue_depth`` (when they are not equal) iterations following
    the ``feed_input`` call.

    The memory location must match the specified ``device`` parameter of the operator.
    For the CPU, the provided memory can be one contiguous buffer or a list of contiguous Tensors.
    For the GPU, to avoid extra copy, the provided buffer must be contiguous. If you provide a list
    of separate Tensors, there will be an additional copy made internally, consuming both memory
    and bandwidth.

    Automatically set to ``True`` when ``parallel=True``

`batch` : bool, optional
    If set to True or None, the ``source`` is expected to produce an entire batch at once.
    If set to False, the ``source`` is called per-sample.

    Setting ``parallel`` to True automatically sets ``batch`` to False if it was not provided.

`batch_info` : bool, optional, default = False
    Controls if a callable ``source`` that accepts an argument and returns batches should receive
    :class:`~nvidia.dali.types.BatchInfo` instance or just an
    integer representing the iteration number.
    If set to False (the default), only the integer is passed. If ``source`` is not callable,
    does not accept arguments or ``batch`` is set to False, setting this flag has no effect.

`parallel` : bool, optional, default = False
    If set to True, the corresponding pipeline will start a pool of Python workers to run the
    callback in parallel. You can specify the number of workers by passing ``py_num_workers``
    into pipeline's constructor.

    When ``parallel`` is set to True, samples returned by ``source`` must be
    NumPy/MXNet/PyTorch CPU arrays or TensorCPU instances.

    |
    Acceptable sources depend on the value specified for ``batch`` parameter.

    If ``batch`` is set to ``False``, the ``source`` must be:

      * a callable (a function or an object with ``__call__`` method) that accepts
        exactly one argument (:class:`~nvidia.dali.types.SampleInfo` instance
        that represents the index of the requested sample).

    If ``batch`` is set to ``True``, the ``source`` can be either:

      * a callable that accepts exactly one argument (either :class:`~nvidia.dali.types.BatchInfo`
        instance or an integer - see ``batch_info`` for details)
      * an iterable,
      * a generator function.

    |
    .. warning::
        Irrespective of ``batch`` value, callables should be stateless - they should produce
        requested sample or batch solely based on the
        :class:`~nvidia.dali.types.SampleInfo`/:class:`~nvidia.dali.types.BatchInfo`
        instance or index in batch, so that they can be run in parallel in a number of workers.

        The ``source`` callback must raise a ``StopIteration`` when the end of the data is reached.
        Note, that due to prefetching, the callback may be invoked with a few iterations past
        the end of dataset - make sure it consistently raises a ``StopIteration`` in that case.

    |
    .. note::
        Callable ``source`` can be run in parallel by multiple workers.
        For ``batch=True`` multiple batches can be prepared in parallel, with ``batch=False``
        it is possible to parallelize computation within the batch.

        When ``batch=True``, callables performance might especially benefit from increasing
        ``prefetch_queue_depth`` so that a few next batches can be computed in parallel.

    |
    .. note::
        Iterator or generator function will be assigned to a single worker that will
        iterate over them. The main advantage is execution in parallel to the main
        Python process, but due to their state it is not possible to calculate more
        than one batch at a time.

`prefetch_queue_depth` : int, option, default = 1
    When run in ``parallel=True`` mode, specifies the number of batches to be computed in
    advance and stored in the internal buffer, otherwise parameter is ignored.
"""

    def __init__(self, source=None, num_outputs=None, *, cycle=None, layout=None, dtype=None,
                 ndim=None, name=None, device="cpu", cuda_stream=None, use_copy_kernel=None,
                 batch=None, parallel=None, no_copy=None, prefetch_queue_depth=None,
                 batch_info=None, **kwargs):
        self._schema = _b.GetSchema("ExternalSource")
        self._spec = _b.OpSpec("ExternalSource")
        self._device = device
        self._layout = layout
        self._dtype = dtype
        self._ndim = ndim
        self._cuda_stream = cuda_stream
        self._use_copy_kernel = use_copy_kernel

        import nvidia.dali.ops
        kwargs, self._call_args = nvidia.dali.ops._separate_kwargs(kwargs)

        callback, source_desc = _get_callback_from_source(source, cycle, batch_info or False)

        if name is not None and num_outputs is not None:
            raise ValueError("`num_outputs` is not compatible with named `ExternalSource`")

        self._name = name
        self._num_outputs = num_outputs
        self._batch = batch
        self._callback = callback
        self._source_desc = source_desc
        self._parallel = parallel
        self._no_copy = no_copy
        self._prefetch_queue_depth = prefetch_queue_depth
        self._batch_info = batch_info

        self._spec.AddArg("device", device)
        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return False

    def __call__(self, *, source=None, cycle=None, name=None, layout=None, dtype=None, ndim=None,
                 cuda_stream=None, use_copy_kernel=None, batch=None, parallel=None, no_copy=None,
                 prefetch_queue_depth=None, batch_info=None, **kwargs):
        ""
        from nvidia.dali.ops import _OperatorInstance
        if batch_info is None:
            batch_info = self._batch_info or False
        elif self._batch_info is not None:
            raise ValueError("The argument ``batch_info`` already specified in constructor.")

        if source is None:
            if cycle is not None:
                if self._callback:
                    raise ValueError(
                        "The argument ``cycle`` can only be specified if ``source`` is an"
                        "iterable object or a generator function specified in this call. "
                        "To cycle through an iterable specified in "
                        "``__init__``, set ``cycle`` there.")
                else:
                    raise ValueError(
                        "The argument ``cycle`` can only be specified if ``source`` is a "
                        "reusable iterable or a generator function.")
            callback = self._callback
            source_desc = self._source_desc
        else:
            if self._callback is not None:
                raise RuntimeError("``source`` already specified in constructor.")
            callback, source_desc = _get_callback_from_source(source, cycle, self._batch_info)

            # Keep the metadata for Pipeline inspection
            self._source_desc = source_desc

        if parallel is None:
            parallel = self._parallel or False
        elif self._parallel is not None:
            raise ValueError("The argument ``parallel`` already specified in constructor.")

        if batch is None:
            batch = self._batch
        elif self._batch is not None:
            raise ValueError("The argument ``batch`` already specified in constructor.")

        # By default parallel is False, so batch will be True
        if batch is None:
            batch = not parallel

        if prefetch_queue_depth is None:
            prefetch_queue_depth = self._prefetch_queue_depth
        elif self._prefetch_queue_depth is not None:
            raise ValueError(
                "The argument ``prefetch_queue_depth`` already specified in constructor.")

        if no_copy is None:
            no_copy = self._no_copy
        elif self._no_copy is not None:
            raise ValueError("The argument ``no_copy`` already specified in constructor.")

        if parallel:
            if prefetch_queue_depth is None:
                prefetch_queue_depth = 1
            if no_copy is None:
                no_copy = True
            if not no_copy:
                raise ValueError("The argument ``no_copy`` cannot be specified to False "
                                 " when used with ``parallel=True``.")
            if prefetch_queue_depth < 1:
                raise ValueError(
                    "``prefetch_queue_depth`` must be a positive integer, got {}.".format(
                        prefetch_queue_depth))
            if source_desc.kind == _SourceKind.CALLABLE:
                if not source_desc.has_inputs:
                    raise TypeError(
                        "Callable passed to External Source in parallel mode "
                        "(when `parallel=True`) must accept exactly one argument: "
                        "`nvidia.dali.types.SampleInfo` if run with `batch=False` or "
                        "either `nvidia.dali.types.BatchInfo` or integer that "
                        "represents the index of the batch within the epoch if `batch=True`. "
                        "Got a callable that does not accept arguments instead.")
            elif not batch:
                if source_desc.kind == _SourceKind.ITERABLE:
                    what = "an iterable"
                else:
                    what = "a generator function"
                raise TypeError(
                    "Parallel external source with {} must be run in a batch mode "
                    "(specify `batch=True` in the external source definition and make sure "
                    "your source returns batches)".format(what))
        else:
            if prefetch_queue_depth is not None:
                raise ValueError("The argument `prefetch_queue_depth` is valid only for "
                                 "parallel external sources (when ``parallel`` is True).")

        if self._layout is not None:
            if layout is not None:
                raise RuntimeError("``layout`` already specified in constructor.")
            else:
                layout = self._layout

        if self._dtype is not None:
            if dtype is not None:
                raise RuntimeError("``dtype`` already specified in constructor.")
            else:
                dtype = self._dtype

        if self._ndim is not None:
            if ndim is not None:
                raise RuntimeError("``ndim`` already specified in constructor.")
            else:
                ndim = self._ndim

        if self._cuda_stream is not None:
            if cuda_stream is not None:
                raise RuntimeError("``cuda_stream`` already specified in constructor.")
            else:
                cuda_stream = self._cuda_stream

        if self._use_copy_kernel is not None:
            if use_copy_kernel is not None:
                raise RuntimeError("``use_copy_kernel`` already specified in constructor.")
            else:
                use_copy_kernel = self._use_copy_kernel

        if name is None:
            name = self._name
        else:
            self._name = name

        if name is not None and self._num_outputs is not None:
            raise RuntimeError("``num_outputs`` is not compatible with named ``ExternalSource``.")

        group_common_kwargs = {
            'cuda_stream': cuda_stream,
            'use_copy_kernel': use_copy_kernel,
            'batch': batch,
            'batch_info': batch_info,
            'parallel': parallel,
            'prefetch_queue_depth': prefetch_queue_depth,
        }

        if self._num_outputs is not None:
            outputs = []
            kwargs = {"no_copy": no_copy}
            group = _ExternalSourceGroup(callback, source_desc, True, **group_common_kwargs)
            for i in range(self._num_outputs):
                if dtype is not None:
                    if isinstance(dtype, (list, tuple)):
                        kwargs['dtype'] = dtype[i] if i < len(
                            dtype) else nvidia.dali.types.DALIDataType.NO_TYPE
                    else:
                        kwargs['dtype'] = dtype
                if ndim is not None:
                    if isinstance(ndim, (list, tuple)):
                        kwargs['ndim'] = ndim[i] if i < len(ndim) else None
                    else:
                        kwargs['ndim'] = ndim
                this_layout = None
                if layout is not None:
                    if isinstance(layout, (list, tuple)):
                        this_layout = layout[i] if i < len(layout) else ""
                    else:
                        this_layout = layout
                    kwargs['layout'] = this_layout
                op_instance = _OperatorInstance([], self, **kwargs)
                op_instance._callback = callback
                op_instance._output_index = i
                op_instance._group = group
                op_instance._layout = this_layout
                op_instance._batch = batch
                group.append(op_instance)
                op_instance.generate_outputs()
                outputs.append(op_instance.unwrapped_outputs)

            return outputs
        else:
            if name is not None:
                kwargs["name"] = name
            if no_copy is not None:
                kwargs["no_copy"] = no_copy
            if dtype is not None:
                kwargs['dtype'] = dtype
            if ndim is not None:
                kwargs['ndim'] = ndim
            if layout is not None:
                kwargs['layout'] = layout
            op_instance = _OperatorInstance([], self, **kwargs)
            op_instance._callback = callback
            op_instance._output_index = None
            op_instance._group = _ExternalSourceGroup(callback, source_desc, False, [op_instance],
                                                      **group_common_kwargs)
            op_instance._layout = layout
            op_instance._batch = batch
            op_instance.generate_outputs()

            return op_instance.unwrapped_outputs

    __doc__ += _args_doc
    __call__.__doc__ += _args_doc


def _is_external_source_with_callback(op_instance):
    return isinstance(op_instance._op, ExternalSource) and op_instance._callback is not None


def _is_external_source(op_instance):
    return isinstance(op_instance._op, ExternalSource)


def _has_external_source(pipeline):
    if not pipeline._py_graph_built:
        pipeline._build_graph()
    for op in pipeline._ops:
        if _is_external_source(op):
            return True
    return False


def external_source(source=None, num_outputs=None, *, cycle=None, name=None, device="cpu",
                    layout=None, dtype=None, ndim=None, cuda_stream=None, use_copy_kernel=None,
                    batch=True, **kwargs):
    """Creates a data node which is populated with data from a Python source.
The data can be provided by the ``source`` function or iterable, or it can be provided by
``pipeline.feed_input(name, data, layout, cuda_stream)`` inside ``pipeline.iter_setup``.

In the case of the GPU input, it is the user responsibility to modify the
provided GPU memory content only using provided stream (DALI schedules a copy on it
and all work is properly queued). If no stream is provided feeding input blocks until the
provided memory is copied to the internal buffer.

.. note::
    :meth:`nvidia.dali.fn.external_source` operator is partially compatible with TensorFlow
    integration via :meth:`nvidia.dali.plugin.tf.experimental.DALIDatasetWithInputs`.
    Please refer to its documentation for details.

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
    """

    from nvidia.dali._debug_mode import _PipelineDebug

    def _external_source(source=None, num_outputs=None, *, cycle=None, name=None, device="cpu",
                         layout=None, dtype=None, ndim=None, cuda_stream=None, use_copy_kernel=None,
                         batch=True, **kwargs):
        if batch is None:
            batch = True

        if num_outputs is not None:
            if source is None:
                raise ValueError(
                    "The parameter ``num_outputs`` is only valid when using ``source`` to "
                    "provide data. To feed multiple external sources in ``feed_input``, "
                    "use multiple ``external_source`` nodes.")

        op = ExternalSource(device=device, num_outputs=num_outputs, source=source, cycle=cycle,
                            layout=layout, dtype=dtype, ndim=ndim, cuda_stream=cuda_stream,
                            use_copy_kernel=use_copy_kernel, batch=batch, **kwargs)
        return op(name=name)

    # Wrapper around external_source to switch between standard and debug mode.
    current_pipeline = _PipelineDebug.current()
    if getattr(current_pipeline, '_debug_on', False):
        return current_pipeline._external_source(
            source=source, num_outputs=num_outputs, cycle=cycle, name=name, device=device,
            layout=layout, batch=batch, **kwargs)
    else:
        return _external_source(source, num_outputs, cycle=cycle, name=name, device=device,
                                layout=layout, dtype=dtype, ndim=ndim, cuda_stream=cuda_stream,
                                use_copy_kernel=use_copy_kernel, batch=batch, **kwargs)


external_source.__doc__ += ExternalSource._args_doc
