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

The convenience function :func:`checkpoint.current` returns the
:class:`~checkpoint.Checkpoint` bound to the current
:class:`EvalContext`. It is created lazily on first access, so the same instance
is reused across the lifetime of the context.

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
