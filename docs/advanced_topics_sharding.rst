Sharding and Shuffling
======================

Sharding
--------

Sharding allows DALI to partition the dataset into non-overlapping pieces on which each DALI pipeline
instance can work. This functionality addresses the issue of having a global and a shared state
that allows the distribution of training samples among the ranks. After each epoch, by default,
the DALI pipeline advances to the next shard to increase the entropy of the data that is seen by
this pipeline. You can alter this behavior by setting the ``stick_to_shard`` reader parameter.

This mode of operation, however, leads to problems when the dataset size is not divisible by the
number of pipelines used or when the shard size is not divisible by the batch size. To address this
issue, and adjust the behavior, you can use the ``pad_last_batch`` reader parameter.

This parameter asks the reader to duplicate the last sample in the last batch of a shard,
which prevents DALI from reading data from the next shard when the batch doesn't divide its size.
The parameter also ensures that all pipelines return the same number of batches, when one batch
is divisible by the batch size but other batches are bigger by one sample. This process pads every
shard to the same size, which is a multiple of the batch size.

Shuffling
---------

Readers provide two kinds of shuffling. ``random_shuffle`` selects samples randomly from a
prefetch buffer and uses the reader's ``seed``. ``shuffle_after_epoch`` changes the dataset or
source-file order between epochs and uses ``shuffle_after_epoch_seed``. When multiple pipelines
partition one dataset, they must use the same ``shuffle_after_epoch_seed`` so that every pipeline
computes the same shuffled ordering before partitioning it into shards.

With ``shuffle_after_epoch=True``, each pipeline keeps its shard index while the shuffled ordering
changes which samples belong to that shard. This conflicts with ``stick_to_shard=True``, which
keeps shard membership stable across epochs, so these options cannot be enabled together.

Whether ``random_shuffle`` can be combined with ``shuffle_after_epoch`` depends on the reader.
See the reader's ``shuffle_after_epoch`` argument documentation for supported combinations.

``random_shuffle`` shuffles the data only within the prefetch buffer. The reader fills the buffer
with ``initial_fill`` samples read sequentially. Each returned sample is then selected randomly
from the buffer and replaced with the next sample from the dataset. When the buffer is much smaller
than the dataset, samples that are far apart in the dataset rarely end up in the same batch.

This matters when the samples are stored in a meaningful order, for example sorted by label.
Readers that work on a list of files, such as ``fn.readers.file``, shuffle the whole list once
when ``random_shuffle`` is enabled, so the order of the files does not matter. Readers of container
formats, such as ``fn.readers.tfrecord``, ``fn.readers.mxnet``, ``fn.readers.caffe``,
``fn.readers.caffe2``, and ``fn.readers.webdataset``, read the samples in the order in which they
are stored. Where such a reader supports ``shuffle_after_epoch``, it changes only the order of the
files, preserving the sample order within each file. Datasets stored in these formats are often
shuffled during creation, so their stored sample order may already be random. If it is meaningful,
shuffle the dataset when you create it or increase ``initial_fill``. A larger buffer uses more
memory and delays the first batch, and a buffer as large as the shard read by the pipeline shuffles
that shard completely.

Framework iterator configuration
--------------------------------

DALI is used in the Deep Learning Frameworks through dedicated iterators, and these iterators need
to be aware of this padding and other reader properties.

Here are the iterator options:

``reader_name``
    Name of the reader operator that provides the iterator's size and last-batch
    padding. It must match the reader's ``name`` argument in every supplied
    pipeline.

    For example, use matching names:

    .. code-block:: python

       @pipeline_def(batch_size=64, num_threads=4, device_id=0)
       def pipeline():
           return fn.readers.file(file_root="/path/to/images", name="train_reader")

       iterator = DALIGenericIterator(
           pipeline(), ["images", "labels"], reader_name="train_reader"
       )

    If a matching, compatible reader is not present in every pipeline, iterator
    construction fails. Providing ``reader_name`` does not change
    ``last_batch_policy``.

    .. tip::

       Prefer ``reader_name`` to setting ``size`` and ``last_batch_padded``
       manually. DALI then keeps the iterator length and padding aligned with the
       reader configuration, including when shards rotate between epochs.

``size``
    Provides the size of the shard for an iterator or, if there is more than one
    shard, the sum of all shard sizes for all wrapped pipelines.

``last_batch_padded``
    Whether the reader pads the last batch by repeating its last sample
    (``True``) or continues into the next epoch (``False``). It applies when
    the shard size is not a multiple of the batch size.

``last_batch_policy``
    Determines the handling of the last batch when the shard size is not divisible
    by the batch size. It affects batches only partially filled with the data. See
    :meth:`~nvidia.dali.plugin.base_iterator.LastBatchPolicy` enum for possible
    values.

``fill_last_batch``
    Deprecated in favour of ``last_batch_policy``. Determines whether the last
    batch should be full, regardless of whether the shard size is divisible by the
    batch size.

Enums
~~~~~

.. autoenum:: nvidia.dali.plugin.base_iterator.LastBatchPolicy
   :members:
   :undoc-members:
   :exclude-members: name

Shard calculation
-----------------

Here is the formula to calculate the shard size for a shard ID::

    floor((id + 1) * dataset_size / num_shards) -
        floor(id * dataset_size / num_shards)

When the pipeline advances through the epochs and the reader moves to the next shard, the formula
needs to be extended to reflect this change::

  floor(((id + epoch_num) % num_shards + 1) * dataset_size / num_shards) -
      floor(((id + epoch_num) % num_shards) * dataset_size / num_shards)

When the second formula is used, providing a size value once at the beginning of the training works
only when the ``stick_to_shard`` reader option is enabled and prevents DALI from rotating shards.
When this occurs, use the first formula.

To address these challenges, use the ``reader_name`` parameter and allow the iterator to
handle the configuration automatically.
