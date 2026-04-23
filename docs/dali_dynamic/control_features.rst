Control Features
================

Dynamic mode chooses sensible defaults for execution mode, thread count,
random state, and device and stream selection that most applications don't
need to override. This section covers the controls that are available when
it makes sense to override them.

**Concepts** walks through each control surface (execution mode, thread
count, random state, device, stream, and evaluation context) and when to
reach for it.

**Tuning an augmentation loop** puts a few of them to work on a small audio
pipeline: execution mode, default-RNG seeding, and per-thread CUDA streams.

.. toctree::
   :maxdepth: 1

   Concepts <control_features_concepts>
   Tuning an augmentation loop <control_features_tutorial>
