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
its constructor and later invoking its `__call__` operator
in :meth:`~nvidia.dali.pipeline.Pipeline.define_graph` method of the Pipeline.

Documentation of every DALI Operator lists **Keyword Arguments** supported by the class constructor.

The documentation for `__call__` operator lists the positional arguments (**Parameters**) and additional **Keyword Arguments**.
`__call__` should be only used in the :meth:`~nvidia.dali.pipeline.Pipeline.define_graph`.
The inputs to the `__call__` operator represent batches of Tensors (TensorLists) processed by DALI,
which are returned by other DALI Operators.

The **Keyword Arguments** listed in `__call__` operator accept TensorList argument inputs. They should be
produced by other 'cpu' Operators.

.. note::
    The names of positional arguments for `__call__` operator (**Parameters**) are only for the
    documentation purposes and should not be used as keyword arguments.

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

Arithmetic expressions
^^^^^^^^^^^^^^^^^^^^^^

DALI allows to use regular Python arithmetic operations within :meth:`~nvidia.dali.pipeline.Pipeline.define_graph`
method on the values returned from invocations of other operators.

The expressions used will be incorporated into the Pipeline without the need to explicitly instantiate operators
and will describe element-wise operations on Tensors.

At least one of the inputs must be a TensorList input that is returned by other DALI Operator.
The other can be :meth:`nvidia.dali.types.Constant` or regular Python value of type `bool`, `int` or `float`.

As the operations performed are element-wise, the shapes of all operands must match.

.. note::
    If one of the operands is a batch of Tensors representing scalars the scalar values
    are *broadcasted* to the other operand.

For details and examples see :doc:`expressions tutorials <examples/general/expressions/index>`.


Supported arithmetic operations
-------------------------------

Currently, DALI supports the following operations:

.. function:: Unary arithmetic operators: +, -

    Unary operators implementing `__pos__(self)` and `__neg__(self)`.
    The result of an unary arithmetic operation always keeps the input type.
    Unary operators accept only TensorList inputs from other operators.

    :rtype: TensorList of the same type

.. function:: Binary arithmetic operations: +, -, *, /, //

    Binary operators implementing `__add__`, `__sub__`, `__mul__`, `__truediv__`
    and `__floordiv__` respectively.

    The result of arithmetic operation between two operands is described below,
    with the exception of `/`, the `__truediv__` operation, which always
    returns `float32` or `float64` types.

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

    `T` stands for any one of the supported numerical types:
    `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`.

    `bool` type is considered the smallest unsigned integer type and is treated as `uint1` with respect
    to the table above.

    .. note::
        Type promotions are commutative.

    .. note::
        The only allowed arithmetic operation between two `bool` values is multiplication `*`.

    :rtype: TensorList of type calculated based on type promotion rules.

.. function:: Comparison operations: ==, !=, <, <=, >, >=

    Comparison operations.

    :rtype: TensorList of `bool` type.

.. function:: Bitwise binary operations: &, |, ^

    The bitwise binary operations abide by the same type promotion rules as arithmetic binary operations,
    but their inputs are restricted to integral types (`bool` included).

    .. note::
        A bitwise operation can be applied to two boolean inputs. Those operations can be used
        to emulate element-wise logical operations on Tensors.

    :rtype: TensorList of type calculated based on type promotion rules.