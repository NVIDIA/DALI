Capture Rules
=============

.. currentmodule:: nvidia.dali.experimental.dynamic

.. role:: python(code)
   :language: python

Capture mode speeds up a loop only for the operators it captures. Uncaptured operators run
eagerly, with no warning and no change in the result.

The capture rule
----------------

Tracing identifies an operator call by its source location and the path of Python calls that
reached it. A helper reached through different call paths is recorded separately for each path.
Captured sites form the pipeline graph.

A call is recorded only when each ordinary input and argument is either:

- The result of another captured call,
- A value that DALI can prove is constant by reading the source, or
- A value wrapped with :func:`capture.invariant`.

On later steps, DALI matches each operator call to a traced call site. Uncaptured and previously
unseen sites run eagerly. At a captured site, a :class:`Batch` argument must be the current output
of the expected captured call. A stale or different batch causes eager fallback. Arguments that
DALI proved constant keep their traced values.

Python does not have to reach every captured call on every step. The fixed pipeline still
computes the call, so putting it behind a condition discards its result rather than saving work.

Captured random calls cannot be skipped because that changes the RNG draw pattern; see
:doc:`random`.

Note that some mismatches are errors instead of eager fallbacks: changing the device, dropping an
:func:`capture.invariant` marker, passing a conflicting batch size, or changing a captured
RNG's draw pattern. See :ref:`capture-limitations`.

Arguments from other calls
--------------------------

A captured call gives a :class:`Batch`, usable as a captured input for the rest of that step.
However, consuming a batch from a previous step leads to falling back to eager execution.

.. code-block:: python

   previous = None
   for jpegs, labels in reader.next_epoch(batch_size=4, capture=True):
       images = ndd.decoders.image(jpegs)
       target = previous if previous is not None else images
       resized = ndd.resize(target, size=[64, 64])   # captured on the first step only
       previous = images                             # stale by the time it is used again

Slicing with :python:`batch.slice[...]`, arithmetic such as :python:`a + b`, and the functions in
:python:`ndd.math` are not captured at all. Each one silently causes dependent operators to fall
back to eager execution, because the result is no longer something the pipeline produced.

Constant arguments
------------------

.. list-table::
   :header-rows: 1
   :widths: 33 55 12

   * - Argument
     - Example
     - Captured
   * - A literal
     - :python:`angle=30`
     - Yes
   * - A list or tuple of constants, written inline
     - :python:`size=[width, 64]`, with :python:`width` a local constant.
     - Yes
   * - A DALI constant
     - :python:`dtype=ndd.int32`
     - Yes
   * - A name assigned once, inside a function, to an immutable value
     - :python:`size = (64, 64)`, then :python:`size=size` as argument.
     - Yes
   * - A constant passed as a function argument
     - :python:`build(images, angle=30)`, where :python:`build` forwards its parameters to an
       operator
     - Yes
   * - Expressions of constants
     - :python:`size=(size, size * 2)`
     - Yes
   * - A name at module level
     - :python:`SIZE = (64, 64)` outside any function.
     - No
   * - A name bound to a mutable value
     - :python:`size = [64, 64]`, then :python:`size=size` as argument.
     - No
   * - A name assigned more than once
     - Including a rebinding on a branch you never take.
     - No
   * - A value returned by a function call
     - :python:`angle = compute_angle()`
     - No
   * - An attribute on your own object
     - :python:`angle=self.angle`, :python:`size=cfg.size`
     - No
   * - A :python:`for` target
     - :python:`for size in sizes:`, then :python:`size=size`
     - No

The rules describe how DALI behaves today. The set of recognized patterns is expected to grow.

A common trap is that a local list is not recognized as constant because DALI cannot prove that
it will not be mutated. Use a tuple instead.

.. code-block:: python

   def not_captured(images):
       size = [64, 64]
       return ndd.resize(images, size=size)

   def captured(images):
       size = (64, 64)
       return ndd.resize(images, size=size)

A list written directly in the call, as in
:python:`ndd.resize(images, size=[64, 64])`, is also recognized as constant.

Proving a constant means reading the call as it appears in the source, so where DALI cannot read
it, constants stop being provable. This is the case of the REPL or code using :python:`exec`.
Arguments that come from other captured calls are unaffected, so such a call still captures if
that is all it takes.

.. note::

   On Python 3.10, DALI cannot differentiate multiple calls on the same line. As a result,
   arguments in a line containing multiple calls cannot be recognized as constants.

   From Python 3.11 onward, several calls on the same line are distinguished using bytecode
   instructions mapped to column offsets as described in :pep:`657`.

For values you cannot express as constants, such as module globals or attributes on a
configuration object, see :ref:`ndd_capture_invariant` below.

.. _ndd_capture_invariant:

The invariant marker
--------------------

:func:`capture.invariant` marks a value that DALI cannot prove constant. Calling it is an
unchecked promise that the value will not change between capture-mode iterations. Module globals
and configuration objects can then participate in captured calls.

It propagates through attribute access, which makes it a good fit for a configuration object
read on every iteration:

.. code-block:: python

   args = parser.parse_args()
   args = ndd.capture.invariant(args)

   for jpegs, labels in reader.next_epoch(batch_size=args.batch_size, capture=True):
       images = ndd.decoders.image(jpegs, device="gpu")
       images = ndd.resize(images, size=args.size)

Without the marker, :python:`args.size` is an attribute on your own object and is not captured.
Marking the namespace once covers every attribute you read from it.

The marker must remain present on later calls. Omitting it raises :python:`RuntimeError`.
Marking part of an expression does not make the rest constant, so
:python:`invariant(0.0) + SOME_GLOBAL` still fails to capture.

API reference
+++++++++++++

.. currentmodule:: nvidia.dali.experimental.dynamic.capture

.. autofunction:: invariant
