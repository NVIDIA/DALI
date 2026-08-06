Random Operators
================

.. currentmodule:: nvidia.dali.experimental.dynamic

Captured random operators execute ahead of Python, so DALI must keep their RNGs aligned with
eager execution.

The draw-pattern rule
---------------------

An RNG used by a captured random operator must make the same draws in the same order on every
iteration. RNGs used only by eager operators are not tracked.

With a fixed draw pattern, capture-mode and eager loops produce the same random values and leave the
RNG in the same state. The guarantee holds across epochs for default and explicit RNGs, regardless
of the source driving the loop.

Allowed and rejected
--------------------

Captured and eager operators may share an RNG. An operator consuming an eager result also runs
eagerly; unrelated operators remain captured.

Changing the draw pattern after tracing raises :class:`RuntimeError`.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Allowed
     - Raises ``RuntimeError``
   * - Sharing one RNG between captured and uncaptured operators
     - Adding a random call that was not there when the loop was traced
   * - A random operator whose arguments cannot be captured, such as a probability read from
       ``self.prob``. It runs eagerly, the rest stays captured
     - Dropping a random call that was traced, since the schedule still expects its draws
   * - Bare ``rng()`` draws in the loop body, which count towards the draw pattern like anything
       else
     - Reseeding or restoring state while the loop is live, including assigning the state it
       already has, because that replaces the generator
   * - Calling the same random site more than once per iteration
     - Touching the RNG between epochs of an :class:`ExternalSource` loop, which re-bases its
       schedule on the generator each epoch

.. dropdown:: Examples

   .. rubric:: Allowed patterns

   Captured and eager calls can share an RNG. Bare draws count towards the fixed pattern. DALI
   warns when a call site repeats and captures only its first call:

   .. code-block:: python

      rng = ndd.random.RNG(seed=42)

      for _ in source.captured(batch_size=4):
          rng()  # bare draw
          eager = ndd.random.coin_flip(batch_size=4, probability=self.prob, rng=rng)
          captured = ndd.random.uniform(batch_size=4, shape=3, rng=rng)

          for repeat in range(2):
              repeated = ndd.random.uniform(batch_size=4, shape=3, rng=rng)

   ``coin_flip`` runs eagerly because ``self.prob`` cannot be proven constant. The other random
   operators stay captured, except for the second call at the repeated site.

   .. rubric:: Adding or dropping a call

   Adding this call after tracing raises:

   .. code-block:: python

      for step, _ in enumerate(source.captured(batch_size=4)):
          ndd.random.uniform(batch_size=4, shape=3, rng=rng)
          if step > 0:
              ndd.random.coin_flip(batch_size=4, rng=rng)

   Dropping this traced call also raises:

   .. code-block:: python

      for step, _ in enumerate(source.captured(batch_size=4)):
          ndd.random.uniform(batch_size=4, shape=3, rng=rng)
          if step == 0:
              ndd.random.coin_flip(batch_size=4, rng=rng)

   .. rubric:: Replacing the RNG state

   Replacing the state while the loop is live raises, even if the assigned state is unchanged:

   .. code-block:: python

      for step, _ in enumerate(source.captured(batch_size=4)):
          if step > 0:
              rng.state = new_state
          ndd.random.uniform(batch_size=4, shape=3, rng=rng)

   .. rubric:: Touching the RNG between epochs

   Changing the RNG between :class:`ExternalSource` epochs raises:

   .. code-block:: python

      source = ndd.ExternalSource(batches, cycle="raise")

      def epoch():
          for _ in source.captured(batch_size=4):
              ndd.random.uniform(batch_size=4, shape=3, rng=rng)

      epoch()
      rng()
      epoch()  # raises RuntimeError

.. warning::

   A random operator that produces a single sample instead of a batch is never captured. That
   happens when it has neither a batch input nor a ``batch_size``, as in
   ``ndd.random.uniform(shape=3, rng=rng)``. The call runs eagerly on every step.

Repeated call sites
-------------------

A random call site may run more than once per iteration. DALI warns and uses the pipeline result
for the first call. Later calls at that site run eagerly because the RNG has already advanced.
Use separate call sites to capture each call.
