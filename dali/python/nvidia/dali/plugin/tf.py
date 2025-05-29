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

import sys

import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from nvidia.dali import internal as _internal

from nvidia.dali.external_source import _is_external_source, _has_external_source
from nvidia.dali.external_source import _is_external_source_with_callback

from nvidia.dali._utils.external_source_impl import _get_generator_from_source_desc
from nvidia.dali._utils.external_source_impl import _cycle_enabled
import nvidia.dali.types as _types

from packaging.version import Version
import warnings

from nvidia.dali_tf_plugin import dali_tf_plugin

from collections.abc import Mapping, Iterable

_dali_tf_module = dali_tf_plugin.load_dali_tf_plugin()
_dali_tf = _dali_tf_module.dali
_dali_tf.__doc__ = (
    _dali_tf.__doc__
    + """

    Please keep in mind that TensorFlow allocates almost all available device memory by default.
    This might cause errors in DALI due to insufficient memory. On how to change this behavior
    please look into the TensorFlow documentation, as it may differ based on your use case.
"""
)

_experimental_dataset_docstring = """Experimental variant of
:class:`~nvidia.dali.plugin.tf.DALIDataset`. This dataset adds support for input tf.data.Datasets.
Support for input tf.data.Datasets is available only for TensorFlow 2.4.1 and newer.

**Input dataset specification**

Each of the input datasets must be mapped to a :meth:`~nvidia.dali.fn.external_source` operator
that will represent the input to the DALI pipeline. In the pipeline the input is represented as
the ``name`` parameter of :meth:`~nvidia.dali.fn.external_source`. Input datasets must be provided
as a mapping from that ``name`` to the dataset object via the `input_datasets` dictionary
argument of DALIDatasetWithInputs.

**Per-sample and batch mode**

The input datasets can operate in per-sample mode or in batch mode.

In per-sample mode, the values produced by the source dataset are interpreted
as individual samples. The batch dimension is absent. For example, a 640x480 RGB image would
have a shape ``[480, 640, 3]``.

In batch mode, the tensors produced by the source dataset are interpreted as batches,
with an additional outer dimension denoting the samples in the batch. For example, a batch of
ten 640x480 RGB images would have a shape ``[10, 480, 640, 3]``.

In both cases (per-sample and batch mode), the layout of those inputs should be denoted as "HWC".

In per-sample mode DALIDataset will query the inputs dataset `batch_size`-times to build a batch
that would be fed into the DALI Pipeline.
In per-sample mode, each sample produced by the input dataset can have a different shape,
but the number of dimension and the layout must remain constant.

**External Source with** ``source`` **parameter**

This experimental DALIDataset accepts pipelines with :meth:`~nvidia.dali.fn.external_source`
nodes that have ``source`` parameter specified.
In that case, the ``source`` will be converted automatically into appropriate
``tf.data.Dataset.from_generator`` dataset with correct placement and
``tf.data.experimental.copy_to_device`` directives.

Those nodes can also work in per-sample or in batch mode. The data in batch mode must be
a dense, uniform tensor (each sample has the same dimensions). Only CPU data is accepted.

This allows TensorFlow DALIDataset to work with most Pipelines that have External Source
``source`` already specified.

.. warning::
    This class is experimental and its API might change without notice.

.. note::
    External source nodes with ``num_outputs`` specified to any number are not
    supported - this means that callbacks with multiple (tuple) outputs are not supported.

.. note::
    External source ``cycle`` policy ``'raise'`` is not supported - the dataset is not restartable.

.. note::
    External source ``cuda_stream`` parameter is ignored - ``source`` is supposed to return
    CPU data and tf.data.Dataset inputs are handled internally.

.. note::
    External source ``use_copy_kernel`` and ``blocking`` parameters are ignored.

.. note::
    Setting ``no_copy`` on the external source nodes when defining the pipeline is considered
    a no-op when used with DALI Dataset. The ``no_copy`` option is handled internally
    and enabled automatically if possible.

.. note::
    Parallel execution of external source callback provided via ``source`` is not supported.
    The callback is executed via TensorFlow ``tf.data.Dataset.from_generator`` - the ``parallel``
    and `prefetch_queue_depth` parameters are ignored.


The operator adds additional parameters to the ones supported by the
:class:`~nvidia.dali.plugin.tf.DALIDataset`:

Parameters
----------
    input_datasets : dict[str, tf.data.Dataset] or
                     dict[str, nvidia.dali.plugin.tf.experimental.Input]
        input datasets to the DALI Pipeline. It must be provided as a dictionary mapping from
        the names of the ``External Source`` nodes to the datasets objects or to the
        :meth:`~nvidia.dali.plugin.tf.experimental.Input` wrapper.

        For example::

            {
                'tensor_input': tf.data.Dataset.from_tensors(tensor).repeat(),
                'generator_input': tf.data.Dataset.from_generator(some_generator)
            }

        can be passed as `input_datasets` for Pipeline like::

            @pipeline_def
            def external_source_pipe():
                input_0 = fn.external_source(name='tensor_input')
                input_1 = fn.external_source(name='generator_input')
                return fn.resize(input_1, resize_x=input_0)

        Entries that use ``tf.data.Dataset`` directly, like::

            {
                'input': tf.data.Dataset.from_tensors(tensor)
            }

        are equivalent to following specification using
        ``nvidia.dali.plugin.tf.experimental.Input``::

            {
                'input' : nvidia.dali.plugin.tf.experimental.Input(
                              dataset=tf.data.Dataset.from_tensors(tensor),
                              layout=None,
                              batch=False)
            }

        This means that inputs, specified as ``tf.data.Dataset`` directly, are considered
        sample inputs.

        .. warning::
            Input dataset must be placed on the same device as ``DALIDatasetWithInputs``.
            If the input has different placement (for instance, input is placed on CPU, while
            ``DALIDatasetWithInputs`` is placed on GPU) the ``tf.data.experimental.copy_to_device``
            with GPU argument must be first applied to input.
"""


_experimental_input_docstring = """Wrapper for an input passed to DALIDataset.
Allows to pass additional options that can override some of the ones specified
in the External Source node in the Python Pipeline object.
Passing None indicates, that the value should be looked up in the pipeline definition.

Parameters
----------
dataset : tf.data.Dataset
    The dataset used as an input
layout : str, optional, default = None
    Layout of the input. If None, the layout will be taken from the corresponding
    External Source node in the Python Pipeline object. If both are provided,
     the layouts must be the same.
    If neither is provided, empty layout will be used.
batch: bool, optional, default = False
    Batch mode of a given input. If None, the batch mode will be taken from the
    corresponding External Source node in the Python Pipeline object.

    If the ``batch = False``, the input dataset is considered sample input.

    If the ``batch = True``, the input dataset is expected to return batches.
"""


def serialize_pipeline(pipeline):
    try:
        return pipeline.serialize()
    except RuntimeError as e:
        raise RuntimeError(
            "Error during pipeline initialization. Note that some operators "
            "(e.g. Python Operators) cannot be used with "
            "TensorFlow Dataset API and DALIIterator."
        ) from e


def DALIIteratorWrapper(
    pipeline=None,
    serialized_pipeline=None,
    sparse=[],
    shapes=[],
    dtypes=[],
    batch_size=-1,
    prefetch_queue_depth=2,
    exec_dynamic=None,
    **kwargs,
):
    """
    TF Plugin Wrapper

    This operator works in the same way as DALI TensorFlow plugin, with the exception that it also
    accepts Pipeline objects as an input, which are serialized internally. For more information,
    see :meth:`nvidia.dali.plugin.tf.DALIRawIterator`.
    """
    if type(prefetch_queue_depth) is dict:
        exec_separated = True
        cpu_prefetch_queue_depth = prefetch_queue_depth["cpu_size"]
        gpu_prefetch_queue_depth = prefetch_queue_depth["gpu_size"]
        if exec_dynamic:
            raise ValueError("Separated queues are not compatible with the dynamic executor.")
    elif type(prefetch_queue_depth) is int:
        exec_separated = False
        cpu_prefetch_queue_depth = -1  # dummy: wont' be used
        gpu_prefetch_queue_depth = prefetch_queue_depth
        if exec_dynamic is None:
            exec_dynamic = True

    if pipeline is not None and pipeline.exec_dynamic:
        exec_dynamic = True

    if serialized_pipeline is None:
        serialized_pipeline = serialize_pipeline(pipeline)

    # if batch_size is not provided we need to extract if from the shape arg
    if (not isinstance(shapes, Iterable) or len(shapes) == 0) and batch_size == -1:
        raise Exception(
            "shapes and batch_size arguments cannot be empty, "
            "please provide at leas one shape argument element with the BATCH size "
            "or set batch_size"
        )

    if len(sparse) > 0 and sparse[0] and batch_size == -1:
        if isinstance(shapes[0], Iterable) and len(shapes[0]) == 1:
            shapes[0] = (shapes[0][0], 1)
        else:
            shapes[0] = (shapes[0], 1)

    # shapes and dtypes need to take into account that sparse tensor will produce 3 output tensors
    new_dtypes = []
    new_shapes = []
    for i in range(len(dtypes)):
        if i < len(sparse) and sparse[i]:
            # indices type of sparse tensor is tf.int64
            new_dtypes.append(tf.int64)
            new_dtypes.append(dtypes[i])
            # dense shape type of sparse tensor is tf.int64
            new_dtypes.append(tf.int64)
            if len(shapes) > i and len(shapes[i]) > 0:
                new_shapes.append((shapes[i][0], 1))
                new_shapes.append((shapes[i][0]))
            else:
                new_shapes.append(())
                new_shapes.append(())
            new_shapes.append(())
        else:
            new_dtypes.append(dtypes[i])
            if len(shapes) > i:
                new_shapes.append(shapes[i])

    # gpu_prefetch_queue_depth correspond to the global queue depth in the uniform case
    out = _dali_tf(
        serialized_pipeline=serialized_pipeline,
        shapes=new_shapes,
        dtypes=new_dtypes,
        sparse=sparse,
        batch_size=batch_size,
        exec_separated=exec_separated,
        gpu_prefetch_queue_depth=gpu_prefetch_queue_depth,
        cpu_prefetch_queue_depth=cpu_prefetch_queue_depth,
        exec_dynamic=exec_dynamic,
        **kwargs,
    )
    new_out = []
    j = 0
    for i in range(len(dtypes)):
        if i < len(sparse) and sparse[i]:
            new_out.append(
                tf.SparseTensor(indices=out[j], values=out[j + 1], dense_shape=out[j + 2])
            )
            j += 3
        else:
            new_out.append(out[j])
            j += 1
    return new_out


def DALIIterator():
    return DALIIteratorWrapper


# Vanilla raw operator legacy
def DALIRawIterator():
    return _dali_tf


def _get_tf_version():
    return Version(tf.__version__)


MIN_TENSORFLOW_VERSION = Version("1.15")


def dataset_compatible_tensorflow():
    """Returns ``True`` if current TensorFlow version is compatible with DALIDataset."""
    return Version(tf.__version__) >= MIN_TENSORFLOW_VERSION


def dataset_inputs_compatible_tensorflow():
    """Returns ``True`` if the current TensorFlow version is compatible with
    experimental.DALIDatasetWithInputs and input Datasets can be used with DALI.
    """
    return Version(tf.__version__) >= Version("2.4.1")


def dataset_distributed_compatible_tensorflow():
    """Returns ``True`` if the tf.distribute APIs for current TensorFlow version are compatible
    with DALIDataset.
    """
    return Version(tf.__version__) >= Version("2.5.0")


def _get_experimental():
    # TODO(klecki): this is WAR only for experimental module
    current_module = sys.modules[__name__]
    experimental = _internal.get_submodule(current_module, "experimental")
    return experimental


def _insert_experimental_member(member, name):
    experimental_module = _get_experimental()
    member.__module__ = experimental_module
    setattr(experimental_module, name, member)


def _get_external_source_param(input_name, input_value, name_es_map, param_name):
    """Get value of the parameter `param_name` specified for the External Source node
       named `input_name`. It can be specified either via `input_value` or in the op instance
       passed in `name_es_map`.
       Not `None` value in `input_value` overwrites the one specified in the Operator instances.
       Otherwise, the one from pipeline definition (the op instance) is used.

    Parameters
    ----------
    input_name : str
        Name of the input
    input_value : Input, optional
        Description of the input
    name_es_map : dict[str, ExternalSource]
        Mapping from the External Source names to operator nodes.
    param_name : str
        name of the parameter we want to access
    """

    def get_param_from_pipe(input_name, name_es_map, param_name):
        es_op = name_es_map[input_name]
        # Check the OpInstance and the `_op`
        try:
            return getattr(es_op, "_" + param_name)
        except AttributeError:
            return getattr(es_op._op, "_" + param_name, None)

    # We didn't get input through input_datasets
    if input_value is None or getattr(input_value, param_name) is None:
        return get_param_from_pipe(input_name, name_es_map, param_name)
    else:
        return getattr(input_value, param_name)


def _get_signature(dtype, shape):
    # TODO(klecki): Find out how we can use ragged tensors for non-uniform batches
    return tf.TensorSpec(shape=shape, dtype=dtype)


def _get_current_device_spec():
    """Best guess at checking the current device string in eager and graph mode.

    Using callable in `with tf.device(...)` for Graph mode will probably break it.
    The graph in use is assumed to be current default graph.
    """
    if tf.executing_eagerly():
        # We are not using this `tf.device` with `with ...`,
        # so we do not change the context, it returns _EagerDeviceContext
        dummy_context_manager = tf.device(None)
        # Get the eager.context singleton instance for this thread
        context = dummy_context_manager._ctx
        # DeviceSpec
        return context.device_spec
    else:
        # Get the default graf, we assume that it's the one in use
        g = tf.compat.v1.get_default_graph()
        # Get the top element of _UserDeviceSpec stack - `with tf.device()` pushes to the stack
        # in graph mode.
        spec = g._device_function_stack.peek_top_obj()
        # Try to normalize to DeviceSpec
        return tf.DeviceSpec.from_string(spec.display_name)


if dataset_compatible_tensorflow():
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.data.util import structure
    import functools

    def dataset_options():
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        if hasattr(options.experimental_optimization, "autotune"):
            options.experimental_optimization.autotune = False
        else:
            options.autotune.enabled = False

        return options

    class _DALIDatasetV2(dataset_ops.DatasetV2):
        def __init__(
            self,
            pipeline,
            output_dtypes=None,
            output_shapes=None,
            fail_on_device_mismatch=True,
            *,
            input_datasets=None,
            batch_size=1,
            num_threads=4,
            device_id=0,
            exec_separated=False,
            exec_dynamic=False,
            prefetch_queue_depth=2,
            cpu_prefetch_queue_depth=2,
            gpu_prefetch_queue_depth=2,
            dtypes=None,
            shapes=None,
        ):
            output_shapes = self._handle_deprecation(output_shapes, shapes, "shapes")
            output_dtypes = self._handle_deprecation(output_dtypes, dtypes, "dtypes")

            if pipeline.exec_dynamic:
                exec_dynamic = True

            if not self._check_dtypes(output_dtypes, tf.DType):
                raise TypeError(
                    "`output_dtypes` should be provided as single tf.DType value "
                    f"or a tuple of tf.DType values. Got value `{output_dtypes}` "
                    f"of the type `{type(output_dtypes)}`."
                )

            if output_shapes is None:
                output_shapes = nest.map_structure(
                    lambda _: tensor_shape.TensorShape(None), output_dtypes
                )
            else:
                output_shapes = nest.map_structure_up_to(
                    output_dtypes, tensor_shape.as_shape, output_shapes
                )

            if not isinstance(output_dtypes, tuple):
                output_dtypes = (output_dtypes,)
                output_shapes = (output_shapes,)

            output_classes = nest.map_structure(lambda _: tf.Tensor, output_dtypes)

            self._pipeline_instance = pipeline  # keep the live Pipeline object
            self._pipeline_serialized = serialize_pipeline(pipeline)
            self._batch_size = batch_size
            self._num_threads = num_threads
            self._device_id = _types.CPU_ONLY_DEVICE_ID if device_id is None else device_id
            self._exec_separated = exec_separated
            self._exec_dynamic = exec_dynamic
            self._prefetch_queue_depth = prefetch_queue_depth
            self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
            self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
            self._output_shapes = output_shapes
            self._output_dtypes = output_dtypes
            self._fail_on_device_mismatch = fail_on_device_mismatch

            self._setup_inputs(input_datasets)

            self._structure = structure.convert_legacy_structure(
                self._output_dtypes, self._output_shapes, output_classes
            )

            super(_DALIDatasetV2, self).__init__(self._as_variant_tensor())

        def _input_lists_from_input_datasets(self, input_datasets, name_es_map):
            """Extract the input specification from the input_datasets dictionary.

            Validate if the inputs exist in the pipeline and the types are correct

            Returns
            -------
            list, list, list, list
                input_datasets, input_names, input_layouts, input_batched
            """

            if input_datasets is None:
                return [], [], [], []

            def _get_dataset(value):
                if isinstance(value, dataset_ops.DatasetV2):
                    return value
                else:
                    return value.dataset

            in_datasets_list = []
            in_names_list = []
            in_layouts_list = []
            in_batched_list = []

            error_str = (
                "`input_datasets` must be a dictionary that maps input names (the `name` "
                "specified for External Source node in DALI pipeline) to input datasets "
                "objects (`tf.data.Dataset`) or `nvidia.dali.plugin.tf.experimental.Input` wrapper "
                "objects"
            )

            if not isinstance(input_datasets, Mapping):
                raise TypeError(
                    error_str + f", got: `{input_datasets}` of type: "
                    "{type(input_datasets)} instead."
                )

            for input_name, input_value in input_datasets.items():
                # keys are str
                if not isinstance(input_name, str):
                    raise TypeError(
                        error_str + f". Expected the keys (representing the input names) to be of "
                        f"type `str`, got: `{input_name}` of type: "
                        f"{input_name} instead."
                    )

                # values are tf.data.Dataset or Input
                is_dataset_only = isinstance(input_value, dataset_ops.DatasetV2)
                experimental = _get_experimental()
                if not is_dataset_only and not isinstance(input_value, experimental.Input):
                    raise TypeError(
                        error_str + ". Expected the values of the dictionary (representing the "
                        "inputs) to be of type `tf.data.Dataset` or "
                        f"`nvidia.dali.plugin.tf.Input` got: `{input_value}` of type: "
                        f"{type(input_value)} instead."
                    )

                # there is External Source with name equal to `input_name`
                if input_name not in name_es_map.keys():
                    raise ValueError(
                        "Did not find an External Source placeholder node with "
                        f"name='{input_name}' in the provided pipeline - required by "
                        "the name specified in the `input_datasets`. Names of "
                        "available placeholder External Source nodes are: "
                        f"{list(name_es_map.keys())}. "
                        "Placeholder nodes cannot have `source` argument specified."
                    )

                in_names_list.append(input_name)
                in_datasets_list.append(_get_dataset(input_value))

                if is_dataset_only:
                    # Set the defaults used in lookup
                    as_input = experimental.Input(input_value, layout=None, batch=False)
                else:
                    as_input = input_value

                # TODO(klecki): Do we want all Python-only ES parameters to be overridable here?
                layout = _get_external_source_param(input_name, as_input, name_es_map, "layout")
                in_layouts_list.append(layout or "")

                # Batched mode is supported by default
                batched = _get_external_source_param(input_name, as_input, name_es_map, "batch")
                in_batched_list.append(batched if batched is not None else True)

            return in_datasets_list, in_names_list, in_layouts_list, in_batched_list

        def _input_lists_from_source(self, callbacked_es_map):
            # TODO(klecki): Warn about this in the doc.
            # We do it only when the users wants to use ExternalSource with `source` specified,
            # as it has some additional limitations.

            # Capture the device that DALI was placed, as we may need to copy the CPU callbacks
            # to that device.
            dali_device_spec = _get_current_device_spec()
            is_dali_on_gpu = dali_device_spec.device_type == "GPU"

            in_datasets_list = []
            in_names_list = []
            in_layouts_list = []
            in_batched_list = []

            for input_name, external_source in callbacked_es_map.items():
                in_names_list.append(input_name)
                layout = _get_external_source_param(input_name, None, callbacked_es_map, "layout")
                in_layouts_list.append(layout or "")

                # Batched mode is supported by default
                batched = _get_external_source_param(input_name, None, callbacked_es_map, "batch")
                in_batched_list.append(batched if batched is not None else True)

                source_desc = external_source._op._source_desc
                if source_desc.cycle == "raise":
                    raise NotImplementedError(
                        f"External Source node: '{input_name}' got argument "
                        "cycle='raise' which is not supported."
                    )

                # All generator datasets must be placed on CPU.
                with tf.device("/cpu:0"):
                    tf_gen, dtype, shape = _get_generator_from_source_desc(
                        source_desc, self._batch_size, external_source._batch
                    )
                    signature = _get_signature(dtype, shape)
                    dataset = tf.data.Dataset.from_generator(tf_gen, output_signature=signature)
                    if _cycle_enabled(source_desc.cycle):
                        dataset = dataset.repeat()
                    # if DALIDataset was placed on GPU, we need to add the copy targeting
                    # that device (with proper id).
                    if is_dali_on_gpu:
                        dataset = dataset.apply(
                            tf.data.experimental.copy_to_device(dali_device_spec.to_string())
                        )
                    in_datasets_list.append(dataset)

            return in_datasets_list, in_names_list, in_layouts_list, in_batched_list

        def _setup_inputs(self, input_datasets):
            """Verify the input specification and assign it to private members in
            normalized form.
            """

            has_es = _has_external_source(self._pipeline_instance)

            # If no inputs are specified, input handling is no-op
            if input_datasets is None and not has_es:
                self._input_datasets = ()
                self._input_names = ()
                self._input_layouts = ()
                self._input_batched = ()
                return

            self._assert_pipeline_instance()

            # To not check everywhere for None
            if input_datasets is None:
                input_datasets = {}

            name_es_map, callbacked_es_map = self._get_name_es_instance_map()

            inputs_from_dict = self._input_lists_from_input_datasets(input_datasets, name_es_map)

            inputs_from_source = self._input_lists_from_source(callbacked_es_map)

            # Check if someone passed an entry in `input_datasets` for the ES with callback
            if not input_datasets.keys().isdisjoint(callbacked_es_map.keys()):
                overlapped = input_datasets.keys().intersection(callbacked_es_map.keys())
                raise ValueError(
                    "Double specification of External Source input is not allowed. "
                    f"External Source nodes named: `{overlapped}` got inputs specified"
                    " via `input_datasets` DALIDataset argument and ExternalSource "
                    "`source` argument at the same time."
                )

            # We covered all inputs
            non_matched = (
                set(name_es_map.keys()) - set(input_datasets.keys()) - set(callbacked_es_map.keys())
            )
            if len(non_matched) != 0:
                raise ValueError(
                    "Found External Source nodes in the Pipeline, that were not "
                    "assigned any inputs. Nodes without inputs: \n"
                    f"{list(non_matched)}.\nNodes that were assigned inputs:\n"
                    f"{list(input_datasets.keys())}."
                )

            self._input_datasets = tuple(inputs_from_dict[0] + inputs_from_source[0])
            self._input_names = tuple(inputs_from_dict[1] + inputs_from_source[1])
            self._input_layouts = tuple(inputs_from_dict[2] + inputs_from_source[2])
            # Map it to integers, to pass as vector<int> instead of vector<bool> to C++
            self._input_batched = tuple(int(b) for b in inputs_from_dict[3] + inputs_from_source[3])

        def _assert_pipeline_instance(self):
            """Ensure that the pipeline is built, and check if the Python part is available."""
            self._pipeline_instance.build()
            if not self._pipeline_instance._py_graph_built and self._pipeline_instance._built:
                raise ValueError(
                    "Deserialized pipelines cannot be used with `input_datasets`. "
                    "Please provide a pipeline that was created directly in Python "
                    "and not recreated from serialized one."
                )

        def _assert_correct_external_sources(self, external_source):
            """Validate that the external source nodes used are properly configured"""
            if external_source._op._num_outputs is not None:
                raise ValueError(
                    "Found placeholder External Source node (without `source` "
                    "argument) in the Pipeline that was created with `num_outputs` "
                    "`num_outputs` parameter. Only single-output "
                    "(with `num_outputs=None`), named (with `name` argument "
                    "specified) External Source nodes are supported as inputs "
                    "placeholders for DALIDataset integration. "
                    "Alternatively, External Source can be used with `source` "
                    "parameter specified."
                )
            if external_source._op._name is None:
                raise ValueError(
                    "Found placeholder External Source node (without `source` "
                    "argument) in the Pipeline that was not named "
                    "(no `name` argument set). Only single-output "
                    "(with `num_outputs=None`), named (with `name` argument "
                    "specified) External Source nodes are supported as inputs "
                    "placeholders for DALIDataset integration. "
                    "Alternatively, External Source can be used with `source` "
                    "parameter specified."
                )

        def _get_name_es_instance_map(self):
            """Return mappings between name of External Source and the op.

            Returns
            -------
            mapping for placeholders nodes, mapping for nodes with Python source
                Two mappings are returned, separating the placeholder nodes without a `source`
                and nodes that got a `source` parameter.
            """
            name_es = {}
            name_es_with_callback = {}
            for op in self._pipeline_instance._ops:
                if _is_external_source_with_callback(op):
                    # use the internal op name (generated automatically in most cases)
                    name_es_with_callback[op.name] = op
                elif _is_external_source(op):
                    self._assert_correct_external_sources(op)
                    # use the user provided name
                    name_es[op._op._name] = op
            return name_es, name_es_with_callback

        def _check_dtypes(self, values, expected_elem_type):
            """Check whether `values` is instance of `expected_elem_type` or tuple of
             `expected_elem_type`.
            TF doesn't treat list as a nesting type, but as a Tensor.
            """
            if isinstance(values, expected_elem_type):
                return True
            elif isinstance(values, tuple) and all(
                isinstance(elem, expected_elem_type) for elem in values
            ):
                return True
            else:
                return False

        def _handle_deprecation(self, supported_arg, deprecated_arg, name):
            if deprecated_arg is not None:
                if supported_arg is not None:
                    raise ValueError(
                        (
                            f"Usage of `{name}` is deprecated in favor of `output_{name}`. "
                            f"Both arguments were provided, but only `output_{name}` "
                            "should be provided."
                        ).format(name=name)
                    )
                # show only this warning
                warnings.warn(
                    (
                        f"Use of argument `{name}` is deprecated. Please use `output_{name}`"
                        f" instead. `output_{name}` should be provided as a tuple"
                        f" or a single value."
                    ).format(name=name),
                    Warning,
                    stacklevel=2,
                )
                if isinstance(deprecated_arg, list):
                    return tuple(deprecated_arg)
                return deprecated_arg
            else:
                return supported_arg

        @property
        def element_spec(self):
            return self._structure

        @property
        def _element_structure(self):
            return self._structure

        def _inputs(self):
            # Apparently here TF is happy with a list
            return nest.flatten(self._input_datasets)

        def _as_variant_tensor(self):
            return _dali_tf_module.dali_dataset(
                # Experimental dataset inputs
                nest.map_structure(lambda d: d._variant_tensor, self._input_datasets),
                # Description of inputs
                input_names=self._input_names,
                input_layouts=self._input_layouts,
                input_batched=self._input_batched,
                # End of experimental inputs
                pipeline=self._pipeline_serialized,
                batch_size=self._batch_size,
                num_threads=self._num_threads,
                device_id=self._device_id,
                exec_separated=self._exec_separated,
                exec_dynamic=self._exec_dynamic,
                prefetch_queue_depth=self._prefetch_queue_depth,
                cpu_prefetch_queue_depth=self._cpu_prefetch_queue_depth,
                gpu_prefetch_queue_depth=self._gpu_prefetch_queue_depth,
                output_shapes=self._output_shapes,
                output_dtypes=self._output_dtypes,
                fail_on_device_mismatch=self._fail_on_device_mismatch,
            )

    if _get_tf_version() < Version("2.0"):

        class _DALIDatasetImpl(dataset_ops.DatasetV1Adapter):
            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                self._wrapped = _DALIDatasetV2(pipeline, **kwargs)
                super(_DALIDatasetImpl, self).__init__(self._wrapped)

    else:
        _DALIDatasetImpl = _DALIDatasetV2

    _experimental_kwargs = ["input_datasets"]

    class DALIDataset(dataset_ops._OptionsDataset):
        @functools.wraps(_DALIDatasetV2.__init__)
        def __init__(self, pipeline, **kwargs):
            # TODO(klecki): Remove this when we move support for inputs from experimental.
            for disallowed_kwarg in _experimental_kwargs:
                if disallowed_kwarg in kwargs.keys():
                    raise TypeError(
                        f"__init__() got an unexpected keyword argument '{disallowed_kwarg}'. "
                        "Dataset inputs are allowed only in"
                        " 'experimental.DALIDatasetWithInputs'."
                    )
            # We detected External Source nodes in the Pipeline
            if _has_external_source(pipeline):
                raise ValueError(
                    "DALIDataset got a DALI pipeline containing External Source "
                    "operator nodes. External Source nodes can be used to express "
                    "placeholders for tf.data.Dataset inputs to DALI or to run "
                    "user-provided Python code via `source` parameter. Support for "
                    "Dataset inputs and External Source's `source` is allowed only "
                    "in 'experimental.DALIDatasetWithInputs'."
                )

            dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
            super(DALIDataset, self).__init__(dataset_impl, dataset_options())

else:

    class DALIDataset:
        def __init__(
            self,
            pipeline,
            output_dtypes=None,
            output_shapes=None,
            fail_on_device_mismatch=True,
            *,
            batch_size=1,
            num_threads=4,
            device_id=0,
            exec_separated=False,
            exec_dynamic=False,
            prefetch_queue_depth=2,
            cpu_prefetch_queue_depth=2,
            gpu_prefetch_queue_depth=2,
            dtypes=None,
            shapes=None,
        ):
            raise RuntimeError(
                "DALIDataset is not supported for detected version of TensorFlow. "
                "DALIDataset supports versions: 1.15, 2.x family"
            )


if dataset_inputs_compatible_tensorflow():

    def _load_experimental_dataset():
        class DALIDatasetWithInputs(dataset_ops._OptionsDataset):
            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
                super(DALIDatasetWithInputs, self).__init__(dataset_impl, dataset_options())

        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, "DALIDatasetWithInputs")

        class Input:
            def __init__(self, dataset, *, layout=None, batch=False):
                if not isinstance(dataset, dataset_ops.DatasetV2):
                    raise TypeError(
                        (
                            "The inputs specified to DALIDataset must be instances of "
                            "type `tf.data.Dataset` got: `{}` of type: {} instead."
                        ).format(dataset, type(dataset))
                    )
                self.dataset = dataset
                self.layout = layout
                self.batch = batch

        Input.__doc__ = _experimental_input_docstring

        _insert_experimental_member(Input, "Input")

    _load_experimental_dataset()

else:

    def _load_experimental_dataset():
        class DALIDatasetWithInputs:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "experimental.DALIDatasetWithInputs is not supported for "
                    "detected version of TensorFlow. DALIDataset supports "
                    "versions: 2.4.1 and above."
                )

        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, "DALIDatasetWithInputs")

        class Input:
            def __init__(self, *args, **kwargs):
                pass

        Input.__doc__ = _experimental_input_docstring

        _insert_experimental_member(Input, "Input")

    _load_experimental_dataset()

DALIDataset.__doc__ = """Creates a ``DALIDataset`` compatible with
    `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ from a DALI
    pipeline. It supports TensorFlow 1.15 and 2.x family.

    ``DALIDataset`` can be placed on CPU and GPU.

    Please keep in mind that TensorFlow allocates almost all available device memory by default.
    This might cause errors in DALI due to insufficient memory. On how to change this behavior
    please look into the TensorFlow documentation, as it may differ based on your use case.

    .. warning::
       Most TensorFlow Datasets have only CPU variant. To process GPU-placed ``DALIDataset`` by
       other TensorFlow dataset you need to first copy it back to CPU using explicit
       ``tf.data.experimental.copy_to_device`` - roundtrip from CPU to GPU back to CPU would
       probably degrade performance a lot and is thus discouraged.

       Additionally, it is advised to not use datasets like ``repeat()`` or similar after
       ``DALIDataset``, which may interfere with DALI memory allocations and prefetching.

    Parameters
    ----------
    pipeline : :class:`nvidia.dali.Pipeline`
        defining the data processing to be performed.
    output_dtypes: tf.DType or tuple of tf.DType, default = None
        expected output types
    output_shapes: tuple of shapes, optional, default = None
        expected output shapes. If provided, must match arity of the `output_dtypes`.
        When set to None, DALI will infer the shapes on its own.
        Individual shapes can be also set to None or contain None to indicate unknown dimensions.
        If specified must be compatible with shape returned from DALI Pipeline
        and with `batch_size` argument which will be the outermost dimension of returned tensors.
        In case of ``batch_size = 1`` it can be omitted in the shape.
        DALI Dataset will try to match requested shape by squeezing 1-sized dimensions
        from shape obtained from Pipeline.
    fail_on_device_mismatch : bool, optional, default = True
        When set to ``True`` runtime check will be performed to ensure DALI device and TF device
        are both CPU or both GPU. In some contexts this check might be inaccurate. When set to
         ``False`` will skip the check but print additional logs to check the devices. Keep in mind
        that this may allow hidden GPU to CPU copies in the workflow and impact performance.
    batch_size : int, optional, default = 1
        batch size of the pipeline.
    num_threads : int, optional, default = 4
        number of CPU threads used by the pipeline.
    device_id : int, optional, default = 0
        id of GPU used by the pipeline.
        A None value for this parameter means that DALI should not use GPU nor CUDA runtime.
        This limits the pipeline to only CPU operators but allows it to run on any
        CPU capable machine.
    exec_separated : bool, optional, default = False
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
        This flag is incompatible with ``exec_dynamic``.
    exec_dynamic : bool, optional, default = False
        Whether to execute the pipeline with the dynamic executor, which allows flexible mixing
        of CPU and GPU operators and enables aggressive memory reuse.
        This flag is incompatible with `exec_separated`.
    prefetch_queue_depth : int, optional, default = 2
        depth of the executor queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to ``False``.
    cpu_prefetch_queue_depth : int, optional, default = 2
        depth of the executor cpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to ``True``.
    gpu_prefetch_queue_depth : int, optional, default = 2
        depth of the executor gpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with `exec_separated` set to ``True``.

    Returns
    -------
    ``DALIDataset`` object based on DALI pipeline and compatible with ``tf.data.Dataset`` API.

    """

DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
