Execution Model
===============

.. currentmodule:: nvidia.dali.experimental.dynamic

Tracing and running
-------------------

Compiled mode traces the first step of the first epoch while executing it eagerly. DALI checks
each operator call for capture and builds a pipeline from the calls it accepts. This usually
makes the traced step slower than an eager step.

Later steps use the pipeline, which is also reused across epochs. If no calls were captured,
DALI warns and continues eagerly.

Performance
-----------

The pipeline prefetches batches while the caller processes the current one. Its stages can also
overlap across iterations.

Compiled mode replaces most Python operator dispatch with a cheaper call-site lookup. This can
help even when the caller provides little work to overlap.

Sources and feeders
-------------------

The loop is driven by one source, which becomes part of the graph. A reader is moved into the
pipeline itself, which is why a torn-down loop can leave it unusable. An :class:`ExternalSource`
stays put and the pipeline polls it. Breaking out of an :class:`ExternalSource` loop leaves the
source usable, but may discard prefetched batches. The next call to ``compiled()`` traces again.

Any *other* :class:`ExternalSource` called inside the loop body is a feeder. The pipeline polls
it ahead of the body rather than at the point where the call appears, so it has to behave
predictably.

A feeder must follow these rules. Violations raise :class:`RuntimeError`:

First used during the traced step
    An instance first called on a later step was never recorded, so there is nothing for it to
    feed.

Read exactly once, on every step
    The pipeline pulls from it whether or not the loop body does, so a skipped read loses data.

Not shared between compiled loops
    Each loop needs its own instance.

Not exhausted before the loop's own source
    A feeder that runs out first is an error rather than a quiet end of epoch.

.. warning::

   An :class:`ExternalSource` instance locks to the first way it is used. Calling it directly
   once prevents it from ever being used in a compiled loop, and binding it to a compiled loop
   prevents direct calls until that loop is torn down. Both raise :class:`RuntimeError`.

.. _capture-limitations:

Limitations
-----------

Every operator call still goes through Python to look up its call site, and compiled mode leaves
the operators alone, so it fuses nothing and generates no code.

Fixed batch size
    ``compile=True`` requires an explicit ``batch_size``, and every later epoch must use the same
    one. An operator called with a conflicting explicit ``batch_size`` raises.

Fixed device
    Giving an operator a different ``device`` on a later iteration raises. This is one of the few
    argument mismatches that does not fall back to eager execution.

Fixed evaluation context
    A live loop cannot run under a different :class:`EvalContext`.

Fixed mode
    A reader used with ``compile=True`` cannot go back to eager iteration, and one already used
    eagerly cannot switch to compiled. Compiled iteration also counts as batch iteration, so it
    cannot be mixed with sample iteration or direct calls on the same reader.

One compiled loop per thread
    Entering a second one in the same thread raises. A loop is not pinned to the thread that
    created it. See :doc:`../threading` for how dynamic mode behaves across threads generally.

Checkpointing
    A reader cannot use compiled mode and checkpointing together, in either order. See
    :doc:`../checkpointing`.

.. warning::

   If a compiled loop driven by a reader consumes a feeder or has captured random operators,
   breaking out destroys the reader driving it. Those sources have already been advanced for a
   step that never completes, so DALI tears the loop down rather than guess. Every later use of
   the reader fails.
