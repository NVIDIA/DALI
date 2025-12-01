.. _mathematical expressions:

Mathematical Expressions
^^^^^^^^^^^^^^^^^^^^^^^^

DALI allows you to use regular Python :ref:`arithmetic operations <Supported Arithmetic Operations>`
and other :ref:`mathematical functions <Mathematical Functions>` in
the pipeline definition (via :meth:`@pipeline_def <nvidia.dali.pipeline_def>` or within
:meth:`~nvidia.dali.Pipeline.define_graph`) on the values that are returned from invoking
other operators.

The expressions that are used will be incorporated into the pipeline without needing to explicitly
instantiate operators and will describe the element-wise operations on tensors::

    @pipeline_def
    def my_pipeline():
        """Create a pipeline which reads and decodes the images, scales channels by
        broadcasting and clamps the result to [128, 255) range."""
        img_files, _ = fn.readers.file(file_root="image_dir")
        images = fn.decoders.image(img_files, device="mixed")
        red_highlight = images * nvidia.dali.types.Constant(np.float32([1.25, 0.75, 0.75]))
        result = nvidia.dali.math.clamp(red_highlight, 128, 255)
        return result


At least one of the inputs to the arithmetic expression must be returned by other DALI operator -
that is a value of :class:`nvidia.dali.pipeline.DataNode` representing a batch of tensors.
The other input can be :meth:`nvidia.dali.types.Constant` or regular Python value of type ``bool``,
``int``, or ``float``. As the operations performed are element-wise, the shapes of all
operands must be compatible - either match exactly or be :ref:`broadcastable <Broadcasting>`.

For details and examples see :doc:`expressions tutorials <examples/general/expressions/index>`.

.. note::
    Keep in mind to wrap the tensor constants used in mathematical expressions (like NumPy array)
    with the :meth:`nvidia.dali.types.Constant`. If used directly, the operator implementation
    from that tensor's library may be picked up and the result might be undefined. Using the line
    from the previous example, the first two variants are equivalent, while the third will be wrong::
      # Correct approach:
      red_highlight_0 = images *
                        nvidia.dali.types.Constant(np.float32([1.25, 0.75, 0.75]))
      red_highlight_1 = nvidia.dali.types.Constant(np.float32([1.25, 0.75, 0.75])) *
                        images
      # Wrong approach:
      # red_highlight_2 = np.float32([1.25, 0.75, 0.75]) * images


.. _type promotions:

Type Promotion Rules
--------------------

For operations that accept two (or more) arguments, type promotions apply.
The resulting type is calculated in accordance to the table below.

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

For more than two arguments, the resulting type is calculated as a reduction from left to right
- first calculating the result of operating on first two arguments, next between that intermediate
result and the third argument and so on, until we have only the result type left.

Supported Arithmetic Operations
-------------------------------

Currently, DALI supports the following operations:

.. function:: Unary arithmetic operators: +, -

    Unary operators that implement ``__pos__(self)`` and ``__neg__(self)``.
    The result of a unary arithmetic operation always preserves the input type.
    Unary operators accept only TensorList inputs from other operators.

    :rtype: TensorList of the same type

.. function:: Binary arithmetic operations: +, -, *, /, //, **

    Binary operators that implement ``__add__``, ``__sub__``, ``__mul__``, ``__truediv__``,
    ``__floordiv__`` and ``__pow__`` respectively.

    The result of an arithmetic operation between two operands is described
    :ref:`above <type promotions>`, with the exception of ``/``, the ``__truediv__`` operation,
    which always returns ``float32`` or ``float64`` type.

    .. note::
        The only allowed arithmetic operation between two ``bool`` values is multiplication
        ``(*)``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.

.. function:: Comparison operations: ==, !=, <, <=, >, >=

    Comparison operations.

    :rtype: TensorList of ``bool`` type.

.. function:: Bitwise binary operations: &, |, ^

    The bitwise binary operations follow the same type promotion rules as arithmetic binary
    operations, but their inputs are restricted to integral types (including ``bool``).

    .. note::
        A bitwise operation can be applied to two boolean inputs. Those operations can be used
        to emulate element-wise logical operations on Tensors.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.

Broadcasting
------------

The term "broadcasting" refers to how tensors with different shapes are treated in mathematical
expressions. A value from a smaller tensor is "broadcast" so it contributes to multiple output
values. At its simplest, a scalar value is broadcast to all output values. In more complex cases,
the values can be broadcast along some dimensions if one of the operands has size 1 and the other is larger::

                [[D, E],       [[ A+D,  B+E ],
    [[A, B]] +   [F, G],   ==   [ A+F,  B+G ],
                 [H, J]]        [ A+H,  B+J ]]


In the example above, the operands have shapes of (1, 2) and (3, 2). The values from the array
[[A, B]] are broadcast along axis 0. It's possible that both operands are subject to broadcasting
along different dimensions::

                [[D],      [[ A+D,  B+D ],
    [[A, B ]] +  [E],  ==   [ A+E,  B+E ],
                 [F]]       [ A+F,  B+F ]]


In this example, the shapes are (1, 2) and (3, 1) - the first operand is broadcast along axis 0 and
the second is broadcast along axis 1.

Shape extension
===============

For convenience, if the arrays have different number of dimensions, the shapes are padded with outer unit dimensions::

    shape of A == (480, 640, 3)
    shape of B == (3)
    shape of A + B == (480, 640, 3)   # b is treated as if it was shaped (1, 1, 3)


Limitations
===========

The broadcasting operations in DALI can have only limited complexity. When broadcasting, the adjacent axes that need
or do not need broadcasting are grouped. There can be up to six alternating broadcast/non-broadcast groups. Example of
grouping::

    shape of A == a, b, 1, c, d
    shape of B == a, b, e, 1, 1
    grouping dimensions (0..1) and (3..4)
    grouped shapes:
    a*b, 1, c*d
    a*b, e, 1


Mathematical Functions
----------------------

Similarly to arithmetic expressions, one can use selected mathematical functions in the Pipeline
graph definition. They also accept :class:`nvidia.dali.pipeline.DataNode`,
:meth:`nvidia.dali.types.Constant` or regular Python value of type ``bool``, ``int``, or ``float``
as arguments. At least one of the inputs must be the output of other DALI Operator.

.. autofunction:: nvidia.dali.math.abs
.. autofunction:: nvidia.dali.math.fabs
.. autofunction:: nvidia.dali.math.floor
.. autofunction:: nvidia.dali.math.ceil
.. autofunction:: nvidia.dali.math.pow
.. autofunction:: nvidia.dali.math.fpow
.. autofunction:: nvidia.dali.math.min
.. autofunction:: nvidia.dali.math.max
.. autofunction:: nvidia.dali.math.clamp

Exponents and logarithms
========================

.. autofunction:: nvidia.dali.math.sqrt
.. autofunction:: nvidia.dali.math.rsqrt
.. autofunction:: nvidia.dali.math.cbrt
.. autofunction:: nvidia.dali.math.exp
.. autofunction:: nvidia.dali.math.log
.. autofunction:: nvidia.dali.math.log2
.. autofunction:: nvidia.dali.math.log10

Trigonometric Functions
=======================

.. autofunction:: nvidia.dali.math.sin
.. autofunction:: nvidia.dali.math.cos
.. autofunction:: nvidia.dali.math.tan
.. autofunction:: nvidia.dali.math.asin
.. autofunction:: nvidia.dali.math.acos
.. autofunction:: nvidia.dali.math.atan
.. autofunction:: nvidia.dali.math.atan2

Hyperbolic Functions
=======================

.. autofunction:: nvidia.dali.math.sinh
.. autofunction:: nvidia.dali.math.cosh
.. autofunction:: nvidia.dali.math.tanh
.. autofunction:: nvidia.dali.math.asinh
.. autofunction:: nvidia.dali.math.acosh
.. autofunction:: nvidia.dali.math.atanh
