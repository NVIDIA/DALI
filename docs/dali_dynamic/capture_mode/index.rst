Capture Mode
============

.. currentmodule:: nvidia.dali.experimental.dynamic

Dynamic mode executes each operator as Python reaches it. Capture mode traces the first step
of a loop and builds a :doc:`DALI pipeline <../../pipeline>` from the calls it can capture.
The pipeline is reused for the rest of that epoch and for later epochs.

The loop body remains ordinary dynamic mode code. Capture mode reduces the Python overhead of
operator calls and allows prefetching later batches while the caller processes the current one.

.. note::

   Capture mode is independent of :class:`EvalMode`. It changes how DALI executes the loop, while
   :class:`EvalMode` controls when results are evaluated.

Turning it on
-------------

A capture-mode loop can start from a reader or an external source. Only the line that drives it
changes: the body stays ordinary dynamic mode code, and the results are the same as without
capture mode.

.. tab-set::

   .. tab-item:: Reader

      Capture mode is enabled for loops driven by a reader by setting the ``capture`` argument
      in ``next_epoch``. Only a single reader per capture-mode loop is supported for now.

      .. code-block:: python
         :emphasize-lines: 5

         import nvidia.dali.experimental.dynamic as ndd

         reader = ndd.readers.File(file_root=images_dir)

         for jpegs, labels in reader.next_epoch(batch_size=4, capture=True):
             images = ndd.decoders.image(jpegs)
             images = ndd.resize(images, size=[64, 64])
             images = ndd.crop_mirror_normalize(
                 images,
                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 dtype=ndd.float32,
             )

   .. tab-item:: External Source

      :meth:`ExternalSource.captured` iterates the source with capture mode enabled for the loop
      body. The loop ends when the source does.

      .. code-block:: python
         :emphasize-lines: 5

         import nvidia.dali.experimental.dynamic as ndd

         source = ndd.ExternalSource(encoded_jpegs)  # callable returning batches of JPEGs

         for jpegs in source.captured(batch_size=4):
             images = ndd.decoders.image(jpegs)
             images = ndd.resize(images, size=[64, 64])
             images = ndd.crop_mirror_normalize(
                 images,
                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 dtype=ndd.float32,
             )

.. warning::

   DALI captures an operator only when it can prove the call is the same on every iteration.
   Other calls run eagerly without warning, so a correct loop may receive little or no benefit.
   See :doc:`capture`.

In this section
---------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Execution model
      :link: concepts
      :link-type: doc

      How tracing, epochs and teardown work, and what a capture-mode loop fixes for its lifetime.

   .. grid-item-card:: Capture rules
      :link: capture
      :link-type: doc

      Which calls use the pipeline, and how to write arguments so they are captured.

   .. grid-item-card:: Random operators
      :link: random
      :link-type: doc

      The extra rule random operators add, and the reproducibility it buys.

   .. grid-item-card:: Tutorial
      :link: tutorial
      :link-type: doc

      A worked image loop, run in capture mode and timed against the same loop run eagerly.

.. toctree::
   :hidden:

   Execution model <concepts>
   Capture rules <capture>
   Random operators <random>
   Tutorial <tutorial>
