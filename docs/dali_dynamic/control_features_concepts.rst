Concepts
========

.. currentmodule:: nvidia.dali.experimental.dynamic

Execution mode
--------------

Every operator call has two phases: a Python preparation phase
(argument checking, shape inference, and dispatch), and the native
execution of the op itself.

The execution mode controls the boundary between those phases. In
synchronous mode, each op runs to completion before the next call
returns:

.. raw:: html

    <style>
        .graphviz { background: transparent !important; }
    </style>

.. digraph:: sync_timeline
   :align: left
   :caption: Synchronous execution mode

   layout=neato;
   bgcolor="transparent";
   dpi=300;

   node [shape=box, style="filled", fontname="NVIDIA Sans, sans-serif",
         fontsize=11, penwidth=0, fontcolor="white",
         fixedsize=true, width=1.1, height=0.5];

   lbl_c  [label=<<table border="0" cellborder="0" cellspacing="0" cellpadding="0"><tr><td width="76" align="right"><font point-size="10" color="#707070">Main thread</font></td></tr></table>>,
           shape=plaintext, fillcolor=transparent, pos="-0.9,0!"];

   p1 [label="prepare 1", fillcolor="#0174DF", pos="0.55,0!"];
   r1 [label="run 1",     fillcolor="#76B900", pos="1.65,0!"];
   p2 [label="prepare 2", fillcolor="#0174DF", pos="3.05,0!"];
   r2 [label="run 2",     fillcolor="#76B900", pos="4.15,0!"];
   p3 [label="prepare 3", fillcolor="#0174DF", pos="5.55,0!"];
   r3 [label="run 3",     fillcolor="#76B900", pos="6.65,0!"];

Operators run asynchronously by default (:attr:`EvalMode.default` is an alias
of :attr:`EvalMode.eager`): each call submits to a single-worker executor
thread that runs them in order, so while it runs op *N* the caller is free to
prepare op *N+1*. The Python caller only blocks when something reads a result
(DLPack export or NumPy conversion), and exceptions from the executor still
surface at the original call site. Note that evaluation is always
synchronous when an op consumes external data (a NumPy array, DLPack
import, or framework tensor).

.. digraph:: async_timeline
   :align: left
   :caption: Asynchronous execution mode

   layout=neato;
   bgcolor="transparent";
   dpi=300;

   node [shape=box, style="filled", fontname="NVIDIA Sans, sans-serif",
         fontsize=11, penwidth=0, fontcolor="white",
         fixedsize=true, width=1.1, height=0.5];

   lbl_c  [label=<<table border="0" cellborder="0" cellspacing="0" cellpadding="0"><tr><td width="76" align="right"><font point-size="10" color="#707070">Main thread</font></td></tr></table>>,
           shape=plaintext, fillcolor=transparent, pos="-0.9,0.55!"];
   lbl_e  [label=<<table border="0" cellborder="0" cellspacing="0" cellpadding="0"><tr><td width="76" align="right"><font point-size="10" color="#707070">Executor thread</font></td></tr></table>>,
           shape=plaintext, fillcolor=transparent, pos="-0.9,0!"];

   // Pads bbox width to match the sync timeline so max-width scaling is
   // identical and node sizes stay consistent between the two figures.
   spacer [shape=point, style=invis, fixedsize=true, width=0.01, height=0.01, pos="7.3,0!"];

   p1 [label="prepare 1", fillcolor="#0174DF", pos="0.55,0.55!"];
   p2 [label="prepare 2", fillcolor="#0174DF", pos="1.95,0.55!"];
   p3 [label="prepare 3", fillcolor="#0174DF", pos="3.35,0.55!"];

   r1 [label="run 1",     fillcolor="#76B900", pos="1.65,0!"];
   r2 [label="run 2",     fillcolor="#76B900", pos="3.05,0!"];
   r3 [label="run 3",     fillcolor="#76B900", pos="4.45,0!"];

.. Fix for a CSS issue causing the dropdown to be invisible after the figure
.. raw:: html

    <div style="clear:both"></div>

.. dropdown:: Debugging with synchronous modes
   :chevron: down-up

   During development, a synchronous mode is easier to work with. Every
   op completes before the next line of Python runs, which means:

   - Breakpoints in ``pdb`` or an IDE stop at the expected location.
   - Exceptions raise at the call site.

   :attr:`EvalMode.sync_cpu` runs each op's work on the caller thread
   before the call returns. That's enough for debugger ergonomics and
   interactive inspection, which makes it the usual choice during
   development.

   .. code-block:: python

      import nvidia.dali.experimental.dynamic as ndd

      with ndd.EvalMode.sync_cpu:
          images = ndd.decoders.image(jpegs, device="gpu")
          resized = ndd.resize(images, size=(224, 224))

   :attr:`EvalMode.sync_full` additionally waits for GPU work to finish
   before the call returns. Use it when the GPU-side wait is what you
   need.

Thread count
------------

DALI's CPU operators run on an internal thread pool. Its size rarely
needs changing, but when it does, use :func:`set_num_threads`:

.. code-block:: python

   ndd.set_num_threads(8)

Call it once, near the start of the program. It sets the process-wide
default thread count used by every auto-created evaluation context.
:func:`get_num_threads` returns the current value.

Without an explicit :func:`set_num_threads` call, DALI picks a default
by looking at (in order):

1. The ``DALI_NUM_THREADS`` environment variable, if set.
2. The number of CPUs available to the process (see :manpage:`sched_getaffinity(2)`).

.. warning::

   Changing the thread count after evaluation has already started
   forces the default evaluation contexts to rebuild their pools, which
   is expensive. Call :func:`set_num_threads` once at startup.

Random number generation
------------------------

The default RNG is thread-local and seeded via :func:`random.set_seed`:

.. code-block:: python

   ndd.random.set_seed(42)

   angle = ndd.random.uniform(batch_size=8, range=(-10, 10))
   images = ndd.rotate(images, angle=angle)

Seeding with the same value produces the same draws. Every random
operator that doesn't explicitly receive an ``rng=`` argument pulls
from the default RNG.

For finer control, construct an explicit :class:`random.RNG` and pass
it per-op:

.. code-block:: python

   geom_rng = ndd.random.RNG(seed=1)
   color_rng = ndd.random.RNG(seed=2)

   angle = ndd.random.uniform(batch_size=8, range=(-10, 10), rng=geom_rng)
   brightness = ndd.random.uniform(batch_size=8, range=(0.8, 1.2), rng=color_rng)

An explicit RNG is useful for independent random streams, such as
geometric vs. color augmentations that should stay reproducible when
reordered.

.. dropdown:: Shard-aware seeding
   :chevron: down-up

   In distributed training, derive each rank's seed from a global seed
   plus the rank index so that shards see different augmentations:

   .. code-block:: python

      rng = ndd.random.RNG(seed=global_seed + rank)

.. note::

   When working with batches in dynamic mode, every random operator needs an
   explicit ``batch_size=`` argument.

Execution environment
---------------------

DALI automatically picks a CUDA device, stream, and thread pool for
each thread of execution, bundled as a thread-local default
:class:`EvalContext`. Most applications never construct one. The
subsections below cover the overrides: custom device pinning, custom
streams for framework interop, and constructing an :class:`EvalContext`
explicitly.

Evaluation context
~~~~~~~~~~~~~~~~~~

An :class:`EvalContext` bundles:

- a CUDA device,
- a thread pool, and
- a CUDA stream.

Typical reasons to construct one directly:

- **Multi-GPU.** Each worker pins to one GPU by entering a :class:`EvalContext`
  built for that device.
- **Custom stream.** DALI work runs on an explicitly chosen stream.
- **Custom thread pool.** Control the thread count without touching the
  process-wide default.

:meth:`EvalContext.default` returns the thread-local default context.
:meth:`EvalContext.current` returns the innermost ``with`` block's
context on the calling thread, or the default when no block is active.

.. code-block:: python

   with ndd.EvalContext(device_id=3, num_threads=4):
       # GPU operators here run on device 3
       ...

.. seealso::

    :doc:`threading` section for the rule about sharing a context across threads.

CUDA streams
~~~~~~~~~~~~

Use the controls below to share a stream with another library, or to
give each thread its own stream in multithreaded code.

:func:`stream` wraps a PyTorch stream, a ``__cuda_stream__``-exposing
object, or a raw CUDA stream handle; called with no argument it
creates a fresh stream on the current device.

There are two places to set a stream:

- :func:`set_default_stream` / :func:`get_default_stream` configure the
  process-wide default for a given device.
- :func:`set_current_stream` / :func:`get_current_stream` configure the
  calling thread's default context stream on the current device.

.. code-block:: python

   import torch

   ndd.set_current_stream(torch.cuda.current_stream())

   # Subsequent ndd ops on this thread run on the PyTorch current stream.
   images = ndd.decoders.image(jpegs, device="gpu")

.. dropdown:: Switching streams mid-run
   :chevron: down-up

   :func:`set_current_stream` does **not** insert any synchronization
   with work previously scheduled on the old stream. DALI work already
   launched on the old stream is not awaited when a later
   ``set_current_stream(new_stream)`` call moves subsequent work to the
   new stream.

Devices
~~~~~~~

A :class:`Device` binds GPU operators on the calling thread to a CUDA
device. Use it when only device pinning is needed: no custom thread
pool, no custom stream.

.. code-block:: python

    with ndd.Device("gpu", device_id=1):
        # GPU operators inside this block run on device 1.
        images = ndd.decoders.image(jpegs, device="gpu")

Reach for :class:`EvalContext` instead when the thread pool or stream
should also vary per device.

.. seealso:: :doc:`control_features_tutorial` demonstrates some of these concepts on a small audio augmentation.
