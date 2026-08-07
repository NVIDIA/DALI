Data Readers Reference
======================

This page documents the data readers available in DALI Dynamic.

Once you create an instance of a reader, you can use it to iterate over the
data multiple times. To start a new iteration over the data, you can call the
``next_epoch`` method, which will return an iterator that yields data.

If ``next_epoch`` is called without `batch_size` argument specified,
the reader will return individual samples. Otherwise, it will return batches
of the specified size.

A reader keeps whichever of the two is used first. It cannot switch between batches and samples,
and once it has started iterating, the batch size cannot change either.

.. code-block:: python

    import nvidia.dali.experimental.dynamic as ndd

    batch_reader = ndd.readers.File(file_root=images_dir)
    for jpegs, labels in batch_reader.next_epoch(batch_size=16):
        # process the batches

    sample_reader = ndd.readers.File(file_root=images_dir)
    for jpeg, label in sample_reader.next_epoch():
        # process the single sample

:doc:`Capture mode <capture_mode/index>` adds further constraints on a reader that uses it.

The table below lists the available readers.

.. include:: operations/dynamic_readers_table

.. include:: operations/dynamic_readers_autodoc
