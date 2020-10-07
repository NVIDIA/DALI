Supported operations
====================

Here is a list of the supported operations:

- | **CPU** operator means that the operator can be scheduled on the CPU.
  | The outputs of CPU operators may be used as regular inputs and to provide per-sample parameters
    for other operators through tensor arguments.
- **GPU** operator means that the operator can be scheduled on the GPU. Their outputs can only be
  used as
  regular inputs for other GPU operators and pipeline outputs.
- **Mixed** operator means that the operator accepts CPU inputs and produces GPU outputs.
- **Sequences** means that the operator can produce or accept as an input a sequence
  (for example, a video).
- **Volumetric** means that the operator supports 3D data processing.

Reading this guide
^^^^^^^^^^^^^^^^^^^^

DALI operators are used in two steps:

#. Parameterizing the operator with ``__init__``.
#. Invoking the parameterized operator like a function (effectively invoking its ``__call__``
   method) in pipeline's :meth:`~nvidia.dali.pipeline.Pipeline.define_graph` method.

In the documentation for every DALI Operator, see the lists of **Keyword Arguments**
that are supported by the class constructor.

The documentation for ``__call__`` operator lists the positional arguments, (or parameters) and
additional keyword arguments. ``__call__`` should only be used in the
:meth:`~nvidia.dali.pipeline.Pipeline.define_graph`. The inputs to the ``__call__`` method
are :class:`nvidia.dali.pipeline.DataNode` objects, which are symbolic representations of
batches of Tensor.

The **keyword arguments** in ``__call__`` operator accept TensorList argument inputs and should be
produced by other CPU operators.

.. note::
    The names of the positional arguments for ``__call__`` operator (**parameters**) are provided
    only for documentation purposes and cannot be used as keyword arguments.

.. note::
    Some keyword arguments can be listed twice. Once for the class constructor and once for
    ``__call__`` operator. This listing means the arguments can be parametrized during operator
    construction with some Python values or driven by the output from another operator when
    running the pipeline.


Support Table
^^^^^^^^^^^^^

The following table lists all available operators and devices on which they can be executed:

.. include:: op_inclusion

Operators Documentation
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: op_autodoc


nvidia.dali.plugin.pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:


Mathematical expressions
^^^^^^^^^^^^^^^^^^^^^^^^
DALI allows you to use regular Python arithmetic operations and other mathematical functions in
the :meth:`~nvidia.dali.pipeline.Pipeline.define_graph` method on the values that are returned
from invoking other operators. Full documentation can be found in the section dedicated to
:ref:`mathematical expressions`.
