JAX Framework
=================

.. note::

   DALI keeps the buffers it exports through DLPack alive by synchronizing the device from a
   background thread. This invalidates the CUDA graphs that XLA captures for its command buffers,
   which surfaces as ``CUDA_ERROR_STREAM_CAPTURE_INVALIDATED``.

   The examples below disable command buffers with the ``XLA_FLAGS`` environment variable. This is
   necessary when combining DALI with ``jax.jit``.

.. toctree::
   :maxdepth: 2

   jax-getting_started.ipynb
   jax-basic_example.ipynb
   jax-multigpu_example.ipynb
   flax-basic_example.ipynb
   pax-basic_example.ipynb
   t5x-basic_example.ipynb
