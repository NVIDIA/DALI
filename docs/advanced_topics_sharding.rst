Sharding
========

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

Framework iterator configuration
--------------------------------

DALI is used in the Deep Learning Frameworks through dedicated iterators, and these iterators need
to be aware of this padding and other reader properties.

Here are the iterator options:

- | ``reader_name`` - Allows you to provide the name of the reader that drives the iterator and
   provides the necessary parameters.

  .. note::
    We recommend that you use this option, so that the next two options
    (``size`` and ``last_batch_padded``) are obtained automatically from the pipeline configuration.
    If it is used, the ``size`` and ``last_batch_padded`` should not be provided explicitly to
    the iterator.

  | This option is more flexible and accurate and takes into account that shard size for a pipeline
    can differ between epochs when the shards are rotated.
- ``size``: Provides the size of the shard for an iterator or, if there is more than one shard,
  the sum of all shard sizes for all wrapped pipelines.
- | ``last_batch_padded``: Determines whether the tail of the data consists of data from the next
    shard (``False``) or is duplicated dummy data (``True``).
  | It is applicable when the shard size is not a multiple of the batch size,
- | ``last_batch_policy`` - Determines the handling of the last batch when the shard size is not
    divisible by the batch size.
  | It affects batches only partially filled with the data. See
    :meth:`~nvidia.dali.plugin.base_iterator.LastBatchPolicy` enum for possible values..

- ``fill_last_batch`` – (Deprecated in favour of ``last_batch_policy``) Determines whether the last
  batch should be full, regardless of whether the shard size is divisible by the batch size.

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

