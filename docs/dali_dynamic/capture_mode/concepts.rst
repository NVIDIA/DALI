Execution Model
===============

.. currentmodule:: nvidia.dali.experimental.dynamic

Tracing and running
-------------------

Capture mode traces the first step of the first epoch while executing it eagerly. DALI checks
each operator call for capture and builds a pipeline from the calls it accepts. This usually
makes the traced step slower than an eager step.

Later steps use the pipeline, which is also reused across epochs. If no calls were captured,
DALI warns and continues eagerly.

Performance
-----------

The pipeline prefetches batches while the caller processes the current one. Its stages can also
overlap across iterations.

Capture mode replaces most Python operator dispatch with a cheaper call-site lookup. This can
help even when the caller provides little work to overlap.

Sources and feeders
-------------------

The loop is driven by one source, which becomes part of the graph. A reader is moved into the
pipeline itself, which is why a torn-down loop can leave it unusable. An :class:`ExternalSource`
stays put and the pipeline polls it. Breaking out of an :class:`ExternalSource` loop leaves the
source usable, but may discard prefetched batches. The next call to ``captured()`` traces again.

Any *other* :class:`ExternalSource` called inside the loop body is a feeder. The pipeline polls
it ahead of the body rather than at the point where the call appears, so it has to behave
predictably.

A feeder must follow these rules. Violations raise :class:`RuntimeError`:

First used during the traced step
    An instance first called on a later step was never recorded, so there is nothing for it to
    feed.

    .. dropdown:: Example

       ``late_feeder`` is first called after tracing:

       .. code-block:: python

          for step, batch in enumerate(source.captured(batch_size=4)):
              ndd.cast(batch, dtype=ndd.float32)
              if step > 0:
                  ndd.cast(late_feeder(), dtype=ndd.float32)  # raises RuntimeError

Read exactly once, on every step
    The pipeline pulls from it whether or not the loop body does, so a skipped read loses data.

    .. dropdown:: Example

       ``feeder`` is read during tracing and skipped on later steps:

       .. code-block:: python

          for step, batch in enumerate(source.captured(batch_size=4)):
              ndd.cast(batch, dtype=ndd.float32)
              if step == 0:
                  ndd.cast(feeder(), dtype=ndd.float32)

Not shared between capture-mode loops
    Each loop needs its own instance.

    .. dropdown:: Example

       ``shared_feeder`` is already bound when the second loop starts:

       .. code-block:: python

          for batch in first_source.captured(batch_size=4):
              ndd.cast(batch, dtype=ndd.float32)
              ndd.cast(shared_feeder(), dtype=ndd.float32)

          for batch in second_source.captured(batch_size=4):
              ndd.cast(batch, dtype=ndd.float32)
              ndd.cast(shared_feeder(), dtype=ndd.float32)  # raises RuntimeError

Not exhausted before the loop's own source
    A feeder that runs out first is an error rather than a quiet end of epoch.

    .. dropdown:: Example

       ``short_batches`` has fewer batches than ``source``:

       .. code-block:: python

          short_feeder = ndd.ExternalSource(short_batches)

          for batch in source.captured(batch_size=4):
              ndd.cast(batch, dtype=ndd.float32)
              ndd.cast(short_feeder(), dtype=ndd.float32)  # raises when exhausted

.. warning::

   An :class:`ExternalSource` instance locks to the first way it is used. Calling it directly
   once prevents it from ever being used in a capture-mode loop, and binding it to one prevents
   direct calls until that loop is torn down. Both raise :class:`RuntimeError`.

.. _capture-limitations:

Limitations
-----------

Every operator call still goes through Python to look up its call site, and capture mode leaves
the operators alone, so it fuses nothing and generates no code.

Fixed batch size
    ``capture=True`` requires an explicit ``batch_size``, and every later epoch must use the same
    one. An operator called with a conflicting explicit ``batch_size`` raises.

    .. dropdown:: Example

        This snippet raises because ``coin_flip`` uses a different batch size:

        .. code-block:: python

           for jpegs, labels in reader.next_epoch(batch_size=128, capture=True):
               images = ndd.decoders.image(jpegs)
               coin_flip = ndd.random.coin_flip(probability=0.5, batch_size=32)


Fixed device
    Giving an operator a different ``device`` on a later iteration raises. This is one of the few
    argument mismatches that does not fall back to eager execution.

    .. dropdown:: Example

       This snippet raises since all iterations must use the same device as during tracing:

       .. code-block:: python

         for jpegs in source.captured(batch_size=128):
             device = "gpu" if fits_on_device(jpegs) else "cpu"
             images = ndd.decoders.image(jpegs, device=device)

Fixed evaluation context
    A live loop cannot run under a different :class:`EvalContext`.

Fixed mode
    A reader used with ``capture=True`` cannot go back to eager iteration, and one already used
    eagerly cannot switch to capture mode. Capture-mode iteration also counts as batch iteration,
    so it cannot be mixed with sample iteration or direct calls on the same reader.

    .. dropdown:: Example

       This snippet raises because the same reader instance is used in capture and eager mode.

       .. code-block:: python
          :emphasize-lines: 4

          for jpegs, labels in reader.next_epoch(batch_size=128, capture=True):
              ...

          for jpegs, labels in reader.next_epoch(batch_size=128):
              ...

One capture-mode loop per thread
    Entering a second one in the same thread raises. A loop is not pinned to the thread that
    created it. See :doc:`../threading` for how dynamic mode behaves across threads generally.

Checkpointing
    A reader cannot use capture mode and checkpointing together, in either order. See
    :doc:`../checkpointing`.

    .. dropdown:: Example

       The snippet below raises when trying to use a checkpointed reader in capture mode.

       .. code-block:: python

          ckpt = ndd.checkpoint.Checkpoint()
          ckpt.register(reader)

          for jpegs, labels in reader.next_epoch(batch_size=128, capture=True):
              ...

.. warning::

   If a capture-mode loop driven by a reader consumes a feeder or has captured random operators,
   breaking out destroys the reader driving it. Those sources have already been advanced for a
   step that never completes, so DALI tears the loop down rather than guess. Every later use of
   the reader fails.
