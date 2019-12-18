Supported operations
====================
- **CPU** operator means that the operator can be scheduled on the CPU. Outputs of CPU operators may be used as regular inputs and to provide per-sample parameters for other operators through tensor arguments.
- **GPU** operator means that the operator can be scheduled on the GPU. Their outputs may only be used as regular inputs for other GPU operators and pipeline outputs.
- **Mixed** operator means that the operator accepts input on the CPU while producing the output on the GPU.
- **Sequences** means that the operator can work (produce or accept as an input) sequence (video like) kind of input.
- **Volumetric** means that the operator supports 3D data processing.

How to read this doc
^^^^^^^^^^^^^^^^^^^^

DALI Operators are used in two steps - creating the parametrized Operator instance using
its constructor and later invoking its `__call__` operator in `define_graph` method of the Pipeline.

Documentation of every DALI Operator lists **Keyword Arguments** supported by the class constructor.

The documentation for `__call__` operator lists the positional arguments (**Parameters**) and additional **Keyword Arguments**.
`__call__` should be only used in the `define_graph`. The inputs to the `__call__` operator represent
Tensors processed by DALI, which are returned by other DALI Operators.

The **Keyword Arguments** listed in `__call__` operator accept Tensor argument inputs. They should be
produced by other 'cpu' Operators.

.. note::
    The names of positional arguments for `__call__` operator (**Parameters**) are only for the
    documentation purposes and should not be used as positional arguments.

.. note::
    Some **Keyword Arguments** can be listed twice - once for class constructor and once for `__call__` operator.
    This means they can be parametrized during operator construction with some Python values
    or driven by output of other operator when running the pipeline.


Support table
^^^^^^^^^^^^^

.. include:: op_inclusion

Operators documentation
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: nvidia.dali.ops
   :members:
   :special-members: __call__

.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:
