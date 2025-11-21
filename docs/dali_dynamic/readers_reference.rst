Data Readers Reference
======================

This page documents the data readers available in DALI Dynamic.

Once you create an instance of a reader, you can use it to iterate over the
data multiple times. To start a new iteration over the data, you can call the
``next_epoch`` method, which will return an iterator that yields data.

If ``next_epoch`` is called without `batch_size` argument specified,
the reader will return individual samples. Otherwise, it will return batches
of the specified size. The batch size can differ from epoch to epoch.

.. code-block:: python

    import nvidia.dali.experimental.dynamic as ndd
    reader = ndd.readers.File(file_root=images_dir)
    for batch in reader.next_epoch(batch_size=16):
        images_batch, labels_batch = batch
        # process the batch
    
    for batch in reader.next_epoch():
        image, label = batch
        # process the single sample
    
The table below lists the available readers.

.. include:: operations/dynamic_readers_table

.. include:: operations/dynamic_readers_autodoc
