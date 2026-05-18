Checkpointing
=============

.. currentmodule:: nvidia.dali.experimental.dynamic

Dynamic mode pipelines can produce *checkpoints* that capture the state of all
stateful operators - readers and random number generators - so that processing
can be resumed at the captured iteration. Operators that do not maintain
user-observable state (decoders, resizes, normalizations, etc.) are conceptually
stateless and are not part of a checkpoint.

This page describes the two checkpointing layers that DALI Dynamic exposes:

#. A :ref:`manual <ndd_checkpointing_manual>` ``get_state`` / ``set_state``
   interface on individual readers and RNGs.
#. A :ref:`semi-automatic <ndd_checkpointing_semi_automatic>`
   :class:`~checkpoint.Checkpoint` aggregator that collects, serializes, and
   restores the state of a registered set of objects.

.. note::

   A :class:`Reader`'s state can be applied only to a freshly constructed
   reader, *before* its first iteration. The underlying prefetch thread starts
   on the first call into the reader, after which the snapshot queue is locked.
   Calls to :meth:`set_state` after iteration has begun raise a
   :class:`RuntimeError`.

.. _ndd_checkpointing_manual:

Manual checkpointing
--------------------

.. note::
  Manual checkpointing is a an advanced feature. It is a set building blocks
  for higher level systems, including the built-in
  :ref:`semi-automatic <ndd_checkpointing_semi_automatic>` checkpointing.
  It allows fine-grained control over individual reader or RNG
  states, enabling integration with pre-existing checkpoint systems or
  transferring state of compatible objects across process boundary. In typical
  usage scenario, it's more convenent to use the
  :ref:`semi-automatic <ndd_checkpointing_semi_automatic>` checkpointing.

Both :class:`~nvidia.dali.experimental.dynamic.random.RNG` and the readers
exposed via ``ndd.readers.*`` provide ``get_state`` and ``set_state`` methods.
The state object returned by ``get_state`` can be converted to a string with
:func:`str`, and ``set_state`` accepts either the state object or its string
representation:

.. code-block:: python

   import nvidia.dali.experimental.dynamic as ndd

   reader = ndd.readers.File(file_root="...")
   it = reader.next_epoch(batch_size=16)

   # Iterate for a while...
   first = next(it)
   second = next(it)

   # Capture a checkpoint after the second batch.
   reader_state = reader.get_state()
   serialized = str(reader_state)  # safe to write to disk, send over the wire, etc.

   # Later, on a fresh reader:
   resumed = ndd.readers.File(file_root="...")
   resumed.set_state(serialized)
   for batch in resumed.next_epoch(batch_size=16):
       ...  # produces the third batch first

The :class:`~nvidia.dali.experimental.dynamic.random.RNG` interface is symmetric:

.. code-block:: python

   rng = ndd.random.RNG(seed=42)
   rng_state = rng.get_state()
   ...
   rng.set_state(rng_state)

.. _ndd_checkpointing_semi_automatic:

Semi-automatic checkpointing with ``Checkpoint``
------------------------------------------------

The :class:`~checkpoint.Checkpoint` class collects the state of a registered
set of stateful objects, serializes it to a single string, and restores the
state of new objects from that string.

A typical save/restore cycle looks like this:

.. code-block:: python

   import nvidia.dali.experimental.dynamic as ndd

   reader = ndd.readers.File(file_root="...")
   rng = ndd.random.RNG(seed=42)

   ckpt = ndd.checkpoint.Checkpoint()
   ckpt.register(reader, "reader")
   ckpt.register(rng, "rng")

   # ... iterate for some time ...

   ckpt.collect()                          # capture the current state
   ckpt.save("ckpt_{seq:04d}.json")        # writes ckpt_0000.json, ckpt_0001.json, ...

Restoring from disk is the symmetric operation:

.. code-block:: python

   reader = ndd.readers.File(file_root="...")
   rng = ndd.random.RNG()

   ckpt = ndd.checkpoint.Checkpoint()
   ckpt.load("ckpt_{seq:04d}.json")        # picks up the most recent file
   ckpt.register(reader, "reader")          # state applied implicitly here
   ckpt.register(rng, "rng")                # ditto

   for batch in reader.next_epoch(batch_size=16):
       ...

Current checkpoint
^^^^^^^^^^^^^^^^^^

The convenience function :func:`checkpoint.current` returns the
:class:`~checkpoint.Checkpoint` bound to the current :class:`EvalContext`.
This function allows the code hidden behind function calls to use checkpointing
without modifying the API to pass the context explicitly.

Using :func:`checkpoint.current`:

.. code-block:: python

   ckpt = ndd.checkpoint.current()
   ckpt.register(reader, "reader")

Registration semantics
^^^^^^^^^^^^^^^^^^^^^^

:meth:`~checkpoint.Checkpoint.register` accepts an optional ``name`` argument:

* If ``name`` is provided, the entry is stored under that key. Any previous op
  registered under the same key is replaced.
* If ``name`` is omitted, the checkpoint first looks up the op by identity. If
  it is already registered, the existing key is returned. Otherwise, internally
  generated sequential names are used.

When the checkpoint is in *loaded* state and the registered key is present in
the loaded dictionary, the saved state is applied to the op immediately. This
makes the load/restore flow above a single line per op.

Lifecycle flags
^^^^^^^^^^^^^^^

The :attr:`~checkpoint.Checkpoint.is_complete` and
:attr:`~checkpoint.Checkpoint.is_loaded` properties reflect the most recent
operation that populated the state dictionary:

* :meth:`~checkpoint.Checkpoint.collect` sets ``is_complete`` and clears
  ``is_loaded``. New ops cannot be registered (call
  :meth:`~checkpoint.Checkpoint.clear` to reset).
* :meth:`~checkpoint.Checkpoint.deserialize` (and :meth:`~checkpoint.Checkpoint.load`)
  set ``is_loaded`` and clear ``is_complete``. Subsequent
  :meth:`~checkpoint.Checkpoint.register` calls must use keys that exist in the
  loaded state.

Filename patterns
^^^^^^^^^^^^^^^^^

:meth:`~checkpoint.Checkpoint.save` and :meth:`~checkpoint.Checkpoint.load`
take a Python format string with a single ``{seq}`` placeholder. ``save``
substitutes the next free sequence number, ``load`` picks the highest one
matching the pattern on disk. Format specifiers (e.g. ``{seq:04d}``) are
honored.

Manual restore
^^^^^^^^^^^^^^

The state of every registered op is applied implicitly when the op is added to
a loaded checkpoint via :meth:`~checkpoint.Checkpoint.register`. The
:meth:`~checkpoint.Checkpoint.restore` method is the explicit counterpart -
it applies all currently dirty states in one call, and is mostly useful when
the ops were registered *before* the state was supplied (e.g. via
:meth:`~checkpoint.Checkpoint.set_state` or
:meth:`~checkpoint.Checkpoint.deserialize`).

Limitations
^^^^^^^^^^^

A few constraints to keep in mind when using ``Checkpoint``:

* **Readers must opt in.** A :class:`Reader` only supports checkpointing when
  constructed with ``enable_checkpointing=True``; the backend then maintains a
  ``prefetch_queue_depth + 1`` deep snapshot queue, which has a small runtime
  cost. Registering a reader that was not opted in is allowed only if its
  backend has not yet been initialized - the call to
  :meth:`~checkpoint.Checkpoint.register` will then enable checkpointing
  retroactively; otherwise it raises a :class:`RuntimeError`.
* **Compiled mode is not supported.** Calling
  :meth:`Reader.next_epoch <nvidia.dali.experimental.dynamic._ops.Reader.next_epoch>`
  with ``compile=True`` on a reader that has checkpointing enabled (or vice
  versa) raises :class:`NotImplementedError`.
* **Reader state must be applied early.** ``Reader.set_state`` (and any
  buffered state propagated through :meth:`~checkpoint.Checkpoint.register`)
  must run before the reader's first iteration; the prefetch thread cannot
  be restored once it has started.
* **The order of anonymous registrations matters.** When a checkpoint is
  loaded and ops are re-added without explicit names, the same number of ops
  must be registered in the same order as at save time. The count is
  validated, and a stored type tag is checked at apply time - so cross-type
  swaps fail loudly - but registering ops of compatible types in a different
  order is not detected. Prefer named registration when in doubt.
* **Format version is strict.** :meth:`~checkpoint.Checkpoint.deserialize`
  rejects payloads whose ``version`` does not match the current format with a
  :class:`ValueError`; there is no automatic upgrade path.
* **Not thread-safe.** A single :class:`Checkpoint` instance must not be
  accessed concurrently from multiple threads. The
  :class:`EvalContext`-bound checkpoint shared by
  :func:`checkpoint.current` follows the same rule.
* **The default ``EvalContext`` is reused.** ``ndd.checkpoint.current()``
  returns the checkpoint bound to the thread-local default
  :class:`EvalContext`, which lives for the lifetime of the process (or the
  enclosing ``with EvalContext(...):`` block). Registrations accumulate
  across unrelated runs unless you call
  :meth:`~checkpoint.Checkpoint.clear` between them.

API reference
-------------

.. currentmodule:: nvidia.dali.experimental.dynamic

ReaderState
^^^^^^^^^^^
.. autoclass:: nvidia.dali.experimental.dynamic._ops.ReaderState
   :members:

Reader checkpoint methods
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: nvidia.dali.experimental.dynamic._ops.Reader.get_state
.. automethod:: nvidia.dali.experimental.dynamic._ops.Reader.set_state

RNG checkpoint methods
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: nvidia.dali.experimental.dynamic.random.RNG.get_state
.. automethod:: nvidia.dali.experimental.dynamic.random.RNG.set_state

.. currentmodule:: nvidia.dali.experimental.dynamic.checkpoint

Checkpoint
^^^^^^^^^^
.. autoclass:: Checkpoint
   :members:

current
^^^^^^^
.. autofunction:: current
