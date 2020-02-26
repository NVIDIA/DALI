# custom wrappers around ops
from future.utils import with_metaclass
from nvidia.dali.ops import _DaliOperatorMeta, _OperatorInstance, _gpu_ops, _cpu_ops
from nvidia.dali import backend as _b
import itertools

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


class _ExternalSourceGroup(object):
    def __init__(self, callback, is_multioutput, instances = []):
        self.instances = list(instances)  # we need a copy!
        self.is_multioutput = is_multioutput
        self.callback = callback
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
                pipeline.feed_input(op._name, data, op._layout)
        else:
            data = callback_out
            op = self.instances[0]
            pipeline.feed_input(op._name, data, op._layout)

class ExternalSource(with_metaclass(_DaliOperatorMeta, object)):
    """ExternalSource is a special operator which can provide data to DALI pipeline from Python
    using several methods.

    The simplest and preferred way is to specify a `source`, which may be a callable or iterable.

    `source` : callable or iterable
    The source of the data. The source is polled for data (via a call or `next(source)` whenever
    the pipeline needs input for the next iteration. The source can supply one or data batches,
    depending on the value or `num_outputs`. If `num_outputs` is not set, the `source` is expected
    to return a single batch. If it's specified, the data is expected to a tuple or list where each
    element corresponds to respective output of the ExternalSource operator.

    `num_outputs`: int or None
    The number of outputs provided by the operator; only valid when used with `source` argument.
    Determines how to interpret the data returned by `source` (a single TensorList or a list
    thereof).

    `cycle`: bool
    If `True`, the source iterable will be wrapped. Otherwise, StopIteration error wil be raised
    when end of data is reached. Setting this flag to True when `source` is not an iterable is an
    error.

    `name` : str, optional
    The name of the data node - used when feeding the data in `iter_setup`; can be omitted if
    the data is provided by `source`.

    `layout` : str or list/tuple of str:
    If provided, sets the layout of the data. When `num_outputs` > 1, layout can be a list
    containing a distinct layout for each output. If the list has fewer elements than `num_outputs`,
    only the first outputs have the layout set, the reset have it cleared.

    """
    global _cpu_ops
    global _gpu_ops
    _cpu_ops = _cpu_ops.union({'ExternalSource'})
    _gpu_ops = _gpu_ops.union({'ExternalSource'})

    def __init__(self, source = None, num_outputs = None, *, cycle = False, layout = None, name = None, device = "cpu", **kwargs):
        self._schema = _b.GetSchema("_ExternalSource")
        self._spec = _b.OpSpec("_ExternalSource")
        self._device = device
        self._layout = layout

        iterable = False
        if source is not None:
            try:
                iterator = iter(source)
                if cycle:
                    iterator = itertools.cycle(iterator)
                iterable = True
                callback = lambda: next(iterator)
            except TypeError:
                if not callable(source):
                    raise "Source must be iterable or callable"
                callback = source
        else:
            callback = None

        if not iterable and cycle:
            raise ValueError("`cycle` argument is only valid for iterable `source`")

        if name is not None and self.num_outputs is not None:
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

    def __call__(self, *, callback = None, name = None, **kwargs):
        if callback is None:
            callback = self._callback
        else:
            if self._callback is not None:
                raise RuntimeError("The callback for external source was specified twice. "
                    "Callback must not be passed both to constructor and call.")

        if name is None:
            name = self._name

        if name is not None and self._num_outputs is not None:
            raise RuntimeError("`num_outputs` is not compatible with named `ExternalSource`")

        if self._num_outputs is not None:
            outputs = []
            kwargs = {}
            group = _ExternalSourceGroup(callback, True)
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
            op_instance._group = _ExternalSourceGroup(callback, False, [op_instance])
            op_instance._layout = self._layout if self._layout is not None else ""
            op_instance.generate_outputs()

            return op_instance.unwrapped_outputs

def _is_external_source_with_callback(op_instance):
    return isinstance(op_instance._op, ExternalSource) and op_instance._callback is not None
