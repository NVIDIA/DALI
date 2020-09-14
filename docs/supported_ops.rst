Supported operations
====================

Here is a list of the supported operations:

- | **CPU** operator means that the operator can be scheduled on the CPU.
  | The outputs of CPU operators may be used as regular inputs and to provide per-sample parameters
    for other operators through tensor arguments.
- **GPU** operator means that the operator can be scheduled on the GPU. Their outputs can only be
  used as
  regular inputs for other GPU operators and pipeline outputs.
- **Mixed** operator means that the operator accepts input on the CPU while producing the output
  on the GPU.
- **Sequences** means that the operator can produce or accept as an input a sequence, or a video
  type of input.
- **Volumetric** means that the operator supports 3D data processing.

Reading this guide
^^^^^^^^^^^^^^^^^^^^

DALI operators are used in two steps:
1. Parameterizing the operator with ``__init__``.
2. Invoking the parameterized operator like a function (effectively invoking its ``__call__``
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

.. automodule:: nvidia.dali.ops
   :members:
   :special-members: __call__

.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:


.. _arithmetic expressions:

Arithmetic expressions
^^^^^^^^^^^^^^^^^^^^^^

DALI allows you to use regular Python arithmetic operations in
the :meth:`~nvidia.dali.pipeline.Pipeline.define_graph` method on the values that are returned
from invoking other operators.

The expressions that are used will be incorporated into the pipeline without needing to explicitly
instantiate operators and will describe the element-wise operations on Tensors.

At least one of the inputs to the arithmetic expression must be returned by other DALI operator -
that is a value of :class:`nvidia.dali.pipeline.DataNode` representing a batch of tensors.
The other input can be :meth:`nvidia.dali.types.Constant` or regular Python value of type ``bool``,
``int``, or ``float``. As the operations performed are element-wise, the shapes of all
operands must match.

.. note::
    If one of the operands is a batch of Tensors that represent scalars, the scalar values
    are *broadcasted* to the other operand.

For details and examples see :doc:`expressions tutorials <examples/general/expressions/index>`.

Supported arithmetic operations
-------------------------------

Currently, DALI supports the following operations:

.. function:: Unary arithmetic operators: +, -

    Unary operators that implement ``__pos__(self)`` and ``__neg__(self)``.
    The result of an unary arithmetic operation always keeps the input type.
    Unary operators accept only TensorList inputs from other operators.

    :rtype: TensorList of the same type

.. function:: Binary arithmetic operations: +, -, *, /, //

    Binary operators that implement ``__add__``, ``__sub__``, ``__mul__``, ``__truediv__``
    and ``__floordiv__`` respectively.

    The result of the arithmetic operation between two operands is described below,
    with the exception of ``/``, the ``__truediv__`` operation, which always
    returns ``float32`` or ``float64`` types.

     ============== ============== ================== ========================
      Operand Type   Operand Type   Result Type        Additional Conditions
     ============== ============== ================== ========================
      T              T              T
      floatX         T              floatX             where T is not a float
      floatX         floatY         floatZ             where Z = max(X, Y)
      intX           intY           intZ               where Z = max(X, Y)
      uintX          uintY          uintZ              where Z = max(X, Y)
      intX           uintY          int2Y              if X <= Y
      intX           uintY          intX               if X > Y
     ============== ============== ================== ========================

    ``T`` stands for any one of the supported numerical types:
    ``bool``, ``int8``, ``int16``, ``int32``, ``int64``, ``uint8``, ``uint16``,
    ``uint32``, ``uint64``, ``float32``, and ``float64``.

    ``bool`` type is considered the smallest unsigned integer type and is treated as ``uint1``
    with respect to the table above.

    .. note::
        Type promotion is commutative.

    .. note::
        The only allowed arithmetic operation between two ``bool`` values is multiplication
        ``(*)``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.

.. function:: Comparison operations: ==, !=, <, <=, >, >=

    Comparison operations.

    :rtype: TensorList of ``bool`` type.

.. function:: Bitwise binary operations: &, |, ^

    The bitwise binary operations follow the same type promotion type as arithmetic binary
    operations, but their inputs are restricted to integral types (including ``bool``).

    .. note::
        A bitwise operation can be applied to two boolean inputs. Those operations can be used
        to emulate element-wise logical operations on Tensors.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.