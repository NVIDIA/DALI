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

import sys

import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from nvidia.dali import types
from nvidia.dali import internal as _internal

from nvidia.dali.external_source import _is_external_source

from collections import Iterable
from distutils.version import LooseVersion
import warnings

from nvidia.dali_tf_plugin import dali_tf_plugin

from collections.abc import Mapping

_dali_tf_module = dali_tf_plugin.load_dali_tf_plugin()
_dali_tf = _dali_tf_module.dali
_dali_tf.__doc__ = _dali_tf.__doc__ + """

    Please keep in mind that TensorFlow allocates almost all available device memory by default. This might cause errors in
    DALI due to insufficient memory. On how to change this behaviour please look into the TensorFlow documentation, as it may
    differ based on your use case.
"""

_experimental_dataset_docstring = """Experimental variant of
:class:`~nvidia.dali.plugin.tf.DALIDataset`. This dataset adds support for input tf.data.Datasets.

Each of the input datasets must be mapped to a :meth:`~nvidia.dali.fn.external_source` operator
that will represent the input to the DALI pipeline. In the pipeline the input is represented as
the ``name`` parameter of :meth:`~nvidia.dali.fn.external_source`. Input datasets must be provided
as a mapping from that ``name`` to the dataset object via the ``input_datasets`` dictionary argument
of DALIDatasetWithInputs.

The input datasets can operate in sample mode - it means that every input dataset
is treated as if providing individual samples, or in batch mode - in that case the input dataset
must return batches (Tensors with outermost dimension representing the sample index in batch).

In sample mode DALIDataset will query the inputs dataset ``batch_size``-times to build a batch
that would be fed into the DALI Pipeline.
In sample mode, each sample produced by the input dataset can have a different shape,
but not a different number of dimensions.

.. warning::
    This class is experimental and its API might change without notice.

.. note::
    Setting ``no_copy`` on the external source nodes when defining the pipeline is considered
    a no-op when used with DALI Dataset. The ``no_copy`` option is handled internally
    and enabled automatically if possible.

The operator adds additional parameters to the ones supported by the
:class:`~nvidia.dali.plugin.tf.DALIDataset`:

Parameters
----------
    input_datasets : dict[str, tf.data.Dataset] or dict[str, nvidia.dali.plugin.tf.experimental.Input]
        input datasets to the DALI Pipeline. It must be provided as a dictionary mapping from
        the names of the ``External Source`` nodes to the datasets objects or to the
        :meth:`~nvidia.dali.plugin.tf.experimental.Input` wrapper.

        For example::

            {
                'tensor_input': tf.data.Dataset.from_tensors(tensor).repeat(),
                'generator_input': tf.data.Dataset.from_generator(some_generator)
            }

        can be passed as ``input_datasets`` for Pipeline like::

            @pipeline_def
            def external_source_pipe():
                input_0 = fn.external_source(name='tensor_input')
                input_1 = fn.external_source(name='generator_input')
                return fn.resize(input_1, resize_x=input_0)

        Entries that use ``tf.data.Dataset`` directly, like::

            {
                'input': tf.data.Dataset.from_tensors(tensor)
            }

        are equivalent to following specification using ``nvidia.dali.plugin.tf.experimental.Input``::

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


def serialize_pipeline(pipeline):
    try:
        return pipeline.serialize()
    except RuntimeError as e:
        raise RuntimeError("Error during pipeline initialization. Note that some operators "
                           "(e.g. Python Operators) cannot be used with "
                           "tensorflow data set API and DALIIterator.") from e


def DALIIteratorWrapper(pipeline=None,
                        serialized_pipeline=None,
                        sparse=[],
                        shapes=[],
                        dtypes=[],
                        batch_size=-1,
                        prefetch_queue_depth=2,
                        **kwargs):
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
    elif type(prefetch_queue_depth) is int:
        exec_separated = False
        cpu_prefetch_queue_depth = -1  # dummy: wont' be used
        gpu_prefetch_queue_depth = prefetch_queue_depth

    if serialized_pipeline is None:
        serialized_pipeline = serialize_pipeline(pipeline)

    # if batch_size is not provided we need to extract if from the shape arg
    if (not isinstance(shapes, Iterable) or len(shapes) == 0) and batch_size == -1:
        raise Exception(
            'shapes and batch_size arguments cannot be empty, '
            'please provide at leas one shape argument element with the BATCH size or set batch_size'
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
    out = _dali_tf(serialized_pipeline=serialized_pipeline,
                   shapes=new_shapes,
                   dtypes=new_dtypes,
                   sparse=sparse,
                   batch_size=batch_size,
                   exec_separated=exec_separated,
                   gpu_prefetch_queue_depth=gpu_prefetch_queue_depth,
                   cpu_prefetch_queue_depth=cpu_prefetch_queue_depth,
                   **kwargs)
    new_out = []
    j = 0
    for i in range(len(dtypes)):
        if i < len(sparse) and sparse[i]:
            new_out.append(
                tf.SparseTensor(indices=out[j], values=out[j + 1], dense_shape=out[j + 2]))
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
    return LooseVersion(tf.__version__)


MIN_TENSORFLOW_VERSION = LooseVersion('1.15')


def dataset_compatible_tensorflow():
    """Returns ``True`` if current TensorFlow version is compatible with DALIDataset."""
    return LooseVersion(tf.__version__) >= MIN_TENSORFLOW_VERSION


def dataset_distributed_compatible_tensorflow():
    """Returns ``True`` if the tf.distribute APIs for current TensorFlow version are compatible
    with DALIDataset.
    """
    return LooseVersion(tf.__version__) >= LooseVersion('2.5.0')


def _get_experimental():
    # TODO(klecki): this is WAR only for experimental module
    current_module = sys.modules[__name__]
    experimental = _internal.get_submodule(current_module, "experimental")
    return experimental


def _insert_experimental_member(member, name):
    experimental_module = _get_experimental()
    member.__module__ = experimental_module
    setattr(experimental_module, name, member)


def _get_layout_from_pipeline(input_name, name_es_map):
    layout = name_es_map[input_name]._layout
    return layout if layout is not None else ""

def _get_external_source_param(input_name, input_value, name_es_map, param_name):
    def get_param_from_pipe(input_name, name_es_map, param_name):
        es_op = name_es_map[input_name]
        # Check the OpInstance and the `_op`
        try:
            return getattr(es_op, "_" + param_name)
        except AttributeError:
            return getattr(es_op._op, "_" + param_name, None)

    # We didn't get input through input_datasets
    if input_value is None:
        return get_param_from_pipe(input_name, name_es_map, param_name)

    if getattr(input_value, param_name) is None:
        return get_param_from_pipe(input_name, name_es_map, param_name)
    else:
        return getattr(input_value, param_name)


if dataset_compatible_tensorflow():
    from tensorflow.python.framework import ops
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.data.util import structure
    import functools

    def dataset_options():
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False

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
                prefetch_queue_depth=2,
                cpu_prefetch_queue_depth=2,
                gpu_prefetch_queue_depth=2,
                dtypes=None,
                shapes=None):

            output_shapes = self._handle_deprecation(output_shapes, shapes, "shapes")
            output_dtypes = self._handle_deprecation(output_dtypes, dtypes, "dtypes")

            if not self._check_dtypes(output_dtypes, tf.DType):
                raise TypeError(("`output_dtypes` should be provided as single tf.DType value "
                    "or a tuple of tf.DType values. Got value `{}` of type `{}`.") \
                        .format(output_dtypes, type(output_dtypes)))

            if output_shapes is None:
                output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None),
                                                   output_dtypes)
            else:
                output_shapes = nest.map_structure_up_to(output_dtypes, tensor_shape.as_shape,
                                                         output_shapes)

            if not isinstance(output_dtypes, tuple):
                output_dtypes = (output_dtypes, )
                output_shapes = (output_shapes, )

            output_classes = nest.map_structure(lambda _: ops.Tensor, output_dtypes)

            self._pipeline_instance = pipeline  # keep the live Pipeline object
            self._pipeline_serialized = serialize_pipeline(pipeline)
            self._batch_size = batch_size
            self._num_threads = num_threads
            if device_id is None:
                device_id = types.CPU_ONLY_DEVICE_ID
            self._device_id = device_id
            self._exec_separated = exec_separated
            self._prefetch_queue_depth = prefetch_queue_depth
            self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
            self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
            self._output_shapes = output_shapes
            self._output_dtypes = output_dtypes
            self._fail_on_device_mismatch = fail_on_device_mismatch

            self._setup_inputs(input_datasets)

            self._structure = structure.convert_legacy_structure(self._output_dtypes,
                                                                 self._output_shapes,
                                                                 output_classes)

            super(_DALIDatasetV2, self).__init__(self._as_variant_tensor())

        def _setup_inputs(self, input_datasets):
            """Verify the input specification and assign it to private members in
            normalized form."""
            if input_datasets is None:
                # Make them explicitly empty for the internal representations
                self._input_datasets = ()
                self._input_names = ()
                self._input_layouts = ()
                self._input_batched = ()
                return

            self._assert_pipeline_instance()

            name_es_map = self._get_name_es_instance_map()

            input_datasets_list = []
            input_names_list = []
            input_layouts_list = []
            input_batched_list = []

            def _get_dataset(value):
                if isinstance(value, dataset_ops.DatasetV2):
                    return value
                else:
                    return value.dataset

            error_str = (
                "`input_datasets` must be a dictionary that maps input names (the `name` "
                "specified for External Source node in DALI pipeline) to input datasets "
                "objects (`tf.data.Dataset`) or `nvidia.dali.plugin.tf.experimental.Input` wrapper "
                "objects")

            if not isinstance(input_datasets, Mapping):
                raise TypeError(error_str +
                                ", got: `{}` of type: {} instead.".format(input_datasets, type(input_datasets)))

            for input_name, input_value in input_datasets.items():
                # keys are str
                if not isinstance(input_name, str):
                    raise TypeError(error_str + (". Expected the keys (representing the input "
                                                 "names) to be of type `str`, got: `{}` of type: "
                                                 "{} instead.").format(input_name, type(input_name)))

                # values are tf.data.Dataset or Input
                is_dataset_only = isinstance(input_value, dataset_ops.DatasetV2)
                experimental = _get_experimental()
                if not is_dataset_only and not isinstance(input_value, experimental.Input):
                    raise TypeError(error_str + (". Expected the values of the dictionary "
                                                 "(representing the inputs) "
                                                 " to be of type `tf.data.Dataset` or "
                                                 "`nvidia.dali.plugin.tf.Input` got: `{}` of "
                                                 "type: {} instead.").format(input_value, type(input_value)))

                # there is External Source with name equal to `input_name`
                if input_name not in name_es_map.keys():
                    raise ValueError(("Did not find an External Source node with name='{}' in "
                                      "the provided pipeline - required by the name specified "
                                      "in the `input_datasets`. Names of available External "
                                      "Source nodes are: {}.").format(input_name,
                                                                      list(name_es_map.keys())))

                input_names_list.append(input_name)
                input_datasets_list.append(_get_dataset(input_value))

                if is_dataset_only:
                    # Set the defaults used in lookup
                    as_input = experimental.Input(input_value, layout=None, batch=False)
                else:
                    as_input = input_value

                # TODO(klecki): Do we want all Python-only ES parameters to be overridable here?
                layout = _get_external_source_param(input_name, as_input, name_es_map, 'layout')
                input_layouts_list.append(layout if layout is not None else "")

                # Batched mode is supported by default
                batched = _get_external_source_param(input_name, as_input, name_es_map, 'batch')
                input_batched_list.append(batched if batched is not None else True)

            # We covered all inputs
            non_matched = set(name_es_map.keys()) - set(input_datasets.keys())
            if len(non_matched) != 0:
                raise ValueError(("Found External Source nodes in the Pipeline, that were not "
                                  "assigned any inputs. Nodes without inputs: \n{}.\nNodes that "
                                  "were assigned inputs:\n{}.").format(list(non_matched), list(input_datasets.keys())))

            self._input_datasets = tuple(input_datasets_list)
            self._input_names = tuple(input_names_list)
            self._input_layouts = tuple(input_layouts_list)
            # Map it to integers
            self._input_batched = tuple(1 if b else 0 for b in input_batched_list)

        def _assert_pipeline_instance(self):
            """Ensure that the pipeline is built, and check if the Python part is available.
            """
            #
            self._pipeline_instance.build()
            if not self._pipeline_instance._py_graph_built and self._pipeline_instance._built:
                raise ValueError("Deserialized pipelines cannot be used with `input_datasets`. "
                                 "Please provide a pipeline that was created directly in Python "
                                 "and not recreated from serialized one.")

        def _assert_correct_external_sources(self, external_source):
            """Validate that the external source nodes used are properly configured"""
            if external_source._op._num_outputs is not None:
                raise ValueError("Found External Source node in the Pipeline that was "
                                 "created with `num_outputs` parameter. Only single-output "
                                 "(with `num_outputs=None`), named (with `name` argument "
                                 "specified) External Source nodes are supported as inputs "
                                 "for DALIDataset integration.")
            if external_source._op._callback is not None:
                raise NotImplementedError("External Source with `callback` specified as input "
                                          "for DALIDataset are not supported yet.")
            if external_source._op._name is None:
                raise ValueError("Found External Source node in the Pipeline that was "
                                 "not named (no `name` argument set). Only single-output "
                                 "(with `num_outputs=None`), named (with `name` argument "
                                 "specified) External Source nodes are supported as inputs "
                                 "for DALIDataset integration.")

        def _get_name_es_instance_map(self):
            name_es = {}
            for op in self._pipeline_instance._ops:
                if _is_external_source(op):
                    self._assert_correct_external_sources(op)
                    name_es[op._op._name] = op
            return name_es

        def _get_layout_from_pipeline(self, input_name, name_es_map):
            layout = name_es_map[input_name]._layout
            return layout if layout is not None else ""

        def _check_dtypes(self, values, expected_elem_type):
            """Check whether `values` is instance of `expected_elem_type`
            or tuple of `expected_elem_type`. TF doesn't treat list as a nesting type, but as a Tensor.
            """
            if isinstance(values, expected_elem_type):
                return True
            elif isinstance(values, tuple) \
                and all(isinstance(elem, expected_elem_type) for elem in values):
                return True
            else:
                return False

        def _handle_deprecation(self, supported_arg, deprecated_arg, name):
            if deprecated_arg is not None:
                if supported_arg is not None:
                    raise ValueError((
                        "Usage of `{name}` is deprecated in favor of `output_{name}`. "
                        "Both arguments were provided, but only `output_{name}` should be provided."
                    ).format(name=name))
                # show only this warning
                warnings.warn(("Use of argument `{name}` is deprecated. Please use `output_{name}` instead. " \
                    + "`output_{name}` should be provided as a tuple or a single value.").format(name=name),
                    Warning, stacklevel=2)
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
                prefetch_queue_depth=self._prefetch_queue_depth,
                cpu_prefetch_queue_depth=self._cpu_prefetch_queue_depth,
                gpu_prefetch_queue_depth=self._gpu_prefetch_queue_depth,
                output_shapes=self._output_shapes,
                output_dtypes=self._output_dtypes,
                fail_on_device_mismatch=self._fail_on_device_mismatch)

    if _get_tf_version() < LooseVersion('2.0'):

        class _DALIDatasetImpl(dataset_ops.DatasetV1Adapter):
            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                self._wrapped = _DALIDatasetV2(pipeline, **kwargs)
                super(_DALIDatasetImpl, self).__init__(self._wrapped)
    else:
        _DALIDatasetImpl = _DALIDatasetV2

    _experimental_kwargs = ['input_datasets']

    class DALIDataset(dataset_ops._OptionsDataset):
        @functools.wraps(_DALIDatasetV2.__init__)
        def __init__(self, pipeline, **kwargs):
            for disallowed_kwarg in _experimental_kwargs:
                if disallowed_kwarg in kwargs.keys():
                    raise TypeError("__init__() got an unexpected keyword argument '{}'".format(
                        disallowed_kwarg))
            dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
            super(DALIDataset, self).__init__(dataset_impl, dataset_options())

    def _load_experimental_dataset():
        class DALIDatasetWithInputs(dataset_ops._OptionsDataset):
            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
                super(DALIDatasetWithInputs, self).__init__(dataset_impl, dataset_options())

        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, "DALIDatasetWithInputs")

        class Input:
            """Wrapper for input passed to DALIDataset. Allows to pass additional options that
            can override some of the ones provided to the External Source node in the
            Python Pipeline object.

            Parameters
            ----------
            dataset : tf.data.Dataset
                The dataset used as input
            layout : str, optional, default = None
                Layout of given input. If None, the layout will be taken from the corresponding
                External Source node in the Python Pipeline object. If it is not provided there,
                empty layout will be used.
            batch: bool, optional, default = None
                Batch mode of given input. If None, the batch mode will be taken from the
                corresponding External Source node in the Python Pipeline object.

                If the ``batch = False``, the input dataset is considered sample input.
                If the ``batch = True``, the input dataset is expected to return batches.
            """
            def __init__(self, dataset, *, layout=None, batch=False):
                if not isinstance(dataset, dataset_ops.DatasetV2):
                    raise TypeError(
                        ("The inputs specified to DALIDataset must be instances of "
                         "type `tf.data.Dataset` got: `{}` of type: {} instead.").format(
                             dataset, type(dataset)))
                self.dataset = dataset
                self.layout = layout
                self.batch = batch

        _insert_experimental_member(Input, "Input")

    _load_experimental_dataset()

else:

    class DALIDataset:
        def __init__(self,
                     pipeline,
                     output_dtypes=None,
                     output_shapes=None,
                     fail_on_device_mismatch=True,
                     *,
                     batch_size=1,
                     num_threads=4,
                     device_id=0,
                     exec_separated=False,
                     prefetch_queue_depth=2,
                     cpu_prefetch_queue_depth=2,
                     gpu_prefetch_queue_depth=2,
                     dtypes=None,
                     shapes=None):
            raise RuntimeError(
                'DALIDataset is not supported for detected version of TensorFlow.  DALIDataset supports versions: 1.15, 2.x family'
            )

    def _load_experimental_dataset():
        class DALIDatasetWithInputs:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    'experimental.DALIDatasetWithInputs is not supported for detected version of TensorFlow. '
                    + 'DALIDataset supports versions: 1.15, 2.x family')

        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, "DALIDatasetWithInputs")

        class Input:
            def __init__(self, *args, **kwargs):
                pass

        _insert_experimental_member(Input, "Input")

    _load_experimental_dataset()

DALIDataset.__doc__ = """Creates a `DALIDataset` compatible with tf.data.Dataset from a DALI
    pipeline. It supports TensorFlow 1.15 and 2.x family


    Please keep in mind that TensorFlow allocates almost all available device memory by default.
    This might cause errors in DALI due to insufficient memory. On how to change this behaviour
    please look into the TensorFlow documentation, as it may differ based on your use case.

    Parameters
    ----------
    pipeline : :class:`nvidia.dali.Pipeline`
        defining the data processing to be performed.
    output_dtypes: tf.DType or tuple of tf.DType, default = None
        expected output types
    output_shapes: tuple of shapes, optional, default = None
        expected output shapes. If provided, must match arity of the ``output_dtypes``.
        When set to None, DALI will infer the shapes on its own.
        Individual shapes can be also set to None or contain None to indicate unknown dimensions.
        If specified must be compatible with shape returned from DALI Pipeline
        and with ``batch_size`` argument which will be the outermost dimension of returned tensors.
        In case of ``batch_size = 1`` it can be omitted in the shape.
        DALI Dataset will try to match requested shape by squeezing 1-sized dimensions
        from shape obtained from Pipeline.
    fail_on_device_mismatch : bool, optional, default = True
        When set to ``True`` runtime check will be performed to ensure DALI device and TF device are
        both CPU or both GPU. In some contexts this check might be inaccurate. When set to ``False``
        will skip the check but print additional logs to check the devices. Keep in mind that this
        may allow hidden GPU to CPU copies in the workflow and impact performance.
    batch_size : int, optional, default = 1
        batch size of the pipeline.
    num_threads : int, optional, default = 4
        number of CPU threads used by the pipeline.
    device_id : int, optional, default = 0
        id of GPU used by the pipeline.
        A None value for this parameter means that DALI should not use GPU nor CUDA runtime.
        This limits the pipeline to only CPU operators but allows it to run on any CPU capable machine.
    exec_separated : bool, optional, default = False
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    prefetch_queue_depth : int, optional, default = 2
        depth of the executor queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with ``exec_separated`` set to ``False``.
    cpu_prefetch_queue_depth : int, optional, default = 2
        depth of the executor cpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with ``exec_separated`` set to ``True``.
    gpu_prefetch_queue_depth : int, optional, default = 2
        depth of the executor gpu queue. Deeper queue makes DALI more
        resistant to uneven execution time of each batch, but it also
        consumes more memory for internal buffers.
        Value will be used with ``exec_separated`` set to ``True``.

    Returns
    -------
    ``DALIDataset`` object based on DALI pipeline and compatible with ``tf.data.Dataset`` API.

    """

DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__
