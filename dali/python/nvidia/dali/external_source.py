# custom wrappers around ops
from nvidia.dali import backend as _b
import inspect

def _check_data_batch(data, batch_size, layout):
    if isinstance(data, (list, tuple)):
        if len(data) != batch_size:
            raise RuntimeError("The external source callback returned an unexpected batch "
            "size: {} instead of {}".format(len(data), batch_size))
        if len(data) > 0:
            dim = len(data[0].shape)
            for t in data:
                if len(t.shape) != dim:
                    raise RuntimeErrorError("All tensors in a batch must have the same number of dimensions")
            if layout != "" and dim != len(layout):
                raise RuntimeError("The layout '{}' cannot describe {}-dimensional data".format(layout, dim))
    else:
        dim = len(data.shape) - 1
        if data.shape[0] != batch_size:
            raise RuntimeError("The external source returned an unexpected batch "
            "size: {} instead of {}".format(data.shape[0], batch_size))
        if layout != "" and dim != len(layout):
            raise RuntimeError("The layout '{}' cannot describe {}-dimensional data".format(layout, dim))

class _CycleIter():
    def __init__(self, iterable):
        self.source = iterable

    def __iter__(self):
        self.it = iter(self.source)
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source)
            return next(self.it)

class _CycleGenFunc():
    def __init__(self, gen_func):
        self.source = gen_func

    def __iter__(self):
        self.it = iter(self.source())
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.source())
            return next(self.it)

class _ExternalSourceGroup(object):
    def __init__(self, callback, is_multioutput, instances = [], cuda_stream = None):
        self.instances = list(instances)  # we need a copy!
        self.is_multioutput = is_multioutput
        self.callback = callback
        self._cuda_stream = cuda_stream
        if callback is not None:
            if callback.__code__.co_argcount not in [0, 1]:
                raise TypeError("External source callback must be a callable with 0 or 1 argument")
            self.accepts_iter_num = (callback.__code__.co_argcount == 1)
        else:
            self.accepts_iter_num = None

    def append(self, instance):
        self.instances.append(instance)

    def call_and_feed(self, pipeline, current_iter):
        if self.accepts_iter_num:
            callback_out = self.callback(current_iter)
        else:
            callback_out = self.callback()

        if self.is_multioutput:
            for op in self.instances:
                data = callback_out[op._output_index]
                pipeline.feed_input(op._name, data, op._layout, self._cuda_stream)
        else:
            data = callback_out
            op = self.instances[0]
            pipeline.feed_input(op._name, data, op._layout, self._cuda_stream)

def _is_generator_function(x):
    """Checks whether x is a generator function or a callable object
    where __call__ is a generator function"""
    if inspect.isgeneratorfunction(x):
        return True
    if x is None or inspect.isfunction(x):
        return False
    call = getattr(x, "__call__", None)
    return _is_generator_function(call)

def _get_callback_from_source(source, cycle):
    iterable = False
    if source is not None:
        try:
            if cycle:
                if inspect.isgenerator(source):
                    raise TypeError("Cannot cycle through a generator - if the generator is a result "
                        "of calling a generator function, pass that function instead as `source`.")
                if _is_generator_function(source):
                    iterator = iter(_CycleGenFunc(source))
                else:
                    iterator = iter(_CycleIter(source))
            else:
                if _is_generator_function(source):
                    source = source()
                iterator = iter(source)
            iterable = True
            callback = lambda: next(iterator)
        except TypeError as err:
            if "not iterable" not in str(err):
                raise(err)
            if cycle is not None:
                raise ValueError("The argument `cycle` can only be specified if `source` is iterable")
            if not callable(source):
                raise TypeError("Source must be callable, iterable or a parameterless generator function")
            callback = source
    else:
        callback = None

    if not iterable and cycle:
        raise ValueError("`cycle` argument is only valid for iterable `source`")
    return callback

class ExternalSource():
    """ExternalSource is a special operator which can provide data to DALI pipeline from Python
using several methods.

The simplest and preferred way is to specify a `source`, which may be a callable or iterable.

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
"""
    _args_doc = """
Args
----

`source` : callable or iterable
    The source of the data. The source is polled for data (via a call ``source()`` or ``next(source)``
    whenever the pipeline needs input for the next iteration. The source can supply one or more data
    batches, depending on the value of ``num_outputs``. If ``num_outputs`` is not set, the ``source`` is
    expected to return a single batch. If it's specified, the data is expected to a be tuple or list
    where each element corresponds to respective return value of the external_source.
    If the source is a callable and has a positional argument, it is assumed to be the current
    iteration number and consecutive calls will be ``source(0)``, ``source(1)``, etc.
    If the source is a generator function, it is invoked and treated as an iterable - however,
    unlike a generator, it can be used with ``cycle``, in which case the function will be called
    again when the generator reaches end of iteration.
    In the case of the GPU input, it is the user responsibility to modify the
    provided GPU memory content only using provided stream (DALI schedules a copy on it
    and all work is properly queued). If no stream is provided, DALI will use a default, with
    best-effort approach at correctness (see ``cuda_stream`` argument documentation for details).

`num_outputs` : int, optional
    If specified, denotes the number of TensorLists produced by the source function

Keyword Args
------------

`cycle`: bool
    If ``True``, the source will be wrapped. Otherwise, StopIteration will be raised
    when end of data is reached. This flag requires that ``source`` is either a collection, i.e. an
    iterable object where ``iter(source)`` will return a fresh iterator on each call or a
    generator function. In the latter case, the generator function will be called again when more
    data is requested than was yielded by the function.

`name` : str, optional
    The name of the data node - used when feeding the data in ``iter_setup``; can be omitted if
    the data is provided by ``source``.

`layout` : :ref:`layout str<layout_str_doc>` or list/tuple thereof
    If provided, sets the layout of the data. When ``num_outputs > 1``, layout can be a list
    containing a distinct layout for each output. If the list has fewer elements than ``num_outputs``,
    only the first outputs have the layout set, the reset have it cleared.

`cuda_stream` : optional, `cudaStream_t` or an object convertible to `cudaStream_t`, e.g. `cupy.cuda.Stream`, `torch.cuda.Stream`
    The CUDA stream, which is going to be used for copying data to GPU or from a GPU
    source. If not set, best effort will be taken to maintain correctness - i.e. if the data
    is provided as a tensor/array from a recognized library (CuPy, PyTorch), the library's
    current stream is used. This should work in typical scenarios, but advanced use cases
    (and code using unsupported libraries) may still need to supply the stream handle
    explicitly.

    Special values:
      *  0 - use default CUDA stream
      * -1 - use DALI's internal stream

    If internal stream is used, the call to ``feed_input`` will block until the copy to internal
    buffer is complete, since there's no way to synchronize with this stream to prevent
    overwriting the array with new data in another stream.
"""

    def __init__(self, source = None, num_outputs = None, *, cycle = None, layout = None, name = None, device = "cpu", cuda_stream = None, **kwargs):
        self._schema = _b.GetSchema("_ExternalSource")
        self._spec = _b.OpSpec("_ExternalSource")
        self._device = device
        self._layout = layout
        self._cuda_stream = cuda_stream

        callback = _get_callback_from_source(source, cycle)

        if name is not None and num_outputs is not None:
            raise ValueError("`num_outputs` is not compatible with named `ExternalSource`")

        self._name = name
        self._num_outputs = num_outputs
        self._callback = callback

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

    def __call__(self, *, source = None, cycle = None, name = None, layout = None, cuda_stream = None, **kwargs):
        ""
        from nvidia.dali.ops import _OperatorInstance

        if source is None:
            if cycle is not None:
                if self._callback:
                    raise ValueError("The argument `cycle` can only be specified if `source` is an iterable object "
                        "or a generator function specified in this call. To cycle through an iterable specified in "
                        "`__init__`, set `cycle` there.")
                else:
                    raise ValueError("The argument `cycle` can only be specified if `source` is a "
                                     "reusable iterable or a generator function.")
            callback = self._callback
        else:
            if self._callback is not None:
                raise RuntimeError("`source` already specified in constructor.")
            callback = _get_callback_from_source(source, cycle)

        if layout is not None and self._layout is not None:
            raise RuntimeError("`layout` already specified in constructor.")

        if cuda_stream is not None and self._cuda_stream is not None:
            raise RuntimeError("`cuda_stream` already specified in constructor.")

        if name is None:
            name = self._name

        if name is not None and self._num_outputs is not None:
            raise RuntimeError("`num_outputs` is not compatible with named `ExternalSource`")

        if self._num_outputs is not None:
            outputs = []
            kwargs = {}
            group = _ExternalSourceGroup(callback, True, cuda_stream=self._cuda_stream)
            for i in range(self._num_outputs):
                op_instance = _OperatorInstance([], self, **kwargs)
                op_instance._callback = callback
                op_instance._output_index = i
                op_instance._group = group
                if self._layout is not None:
                    if isinstance(self._layout, (list, tuple)):
                        op_instance._layout = self._layout[i] if i < len(self._layout) else ""
                    else:
                        op_instance._layout = self._layout
                else:
                    op_instance._layout = ""

                group.append(op_instance)
                op_instance.generate_outputs()
                outputs.append(op_instance.unwrapped_outputs)

            return outputs
        else:
            if name is not None:
                kwargs["name"] = name
            op_instance = _OperatorInstance([], self, **kwargs)
            op_instance._callback = callback
            op_instance._output_index = None
            op_instance._group = _ExternalSourceGroup(callback, False, [op_instance], cuda_stream=self._cuda_stream)
            op_instance._layout = self._layout if self._layout is not None else ""
            op_instance.generate_outputs()

            return op_instance.unwrapped_outputs

    __doc__ += _args_doc
    __call__.__doc__ += _args_doc


def _is_external_source_with_callback(op_instance):
    return isinstance(op_instance._op, ExternalSource) and op_instance._callback is not None

def external_source(source = None, num_outputs = None, *, cycle = None, name = None, device = "cpu", layout = None, cuda_stream = None):
    """Creates a data node which is populated with data from a Python source.
The data can be provided by the ``source`` function or iterable, or it can be provided by
``pipeline.feed_input(name, data, layout)`` inside ``pipeline.iter_setup``.
    In the case of the GPU input, it is the user responsibility to modify the
    provided GPU memory content only using provided stream (DALI schedules a copy on it
    and all work is properly queued). If no stream is provided feeding input blocks until the
    provided memory is copied to the internal buffer

.. note::
    To return a batch of copies of the same tensor, use :func:`nvidia.dali.types.Constant`,
    which is more performant.
    """
    if num_outputs is not None:
        if source is None:
            raise ValueError("The parameter `num_outputs` is only valid when using `source` to "
                "provide data. To feed multiple external sources in `feed_input`, use multiple "
                "`external_source` nodes.")

    op = ExternalSource(device = device, num_outputs = num_outputs, source = source,
                        cycle = cycle, layout = layout, cuda_stream = cuda_stream)
    return op(name = name)

external_source.__doc__ += ExternalSource._args_doc
