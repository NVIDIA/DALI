Advanced Topics
=================

This section covers a few advanced topics that are mentioned in the API documentation.

.. note::
  For typical use cases, the default DALI configuration performs well out of the box, and you do
  not need to review this section.

Thread Affinity
---------------

This functionality allows you to pin DALI threads to the specified CPU. Thread affinity avoids
the overhead of worker threads jumping from core to core and improves performance with CPU-heavy
workloads. You can set the DALI CPU thread affinity by using the ``DALI_AFFINITY_MASK`` environment
variable, which is a comma-separated list of CPU IDs that will be assigned to corresponding DALI
threads. The number of DALI threads is set during the pipeline construction by the ``num_threads``
argument and ``set_affinity`` enables thread affinity for the CPU worker threads.

.. note::
  For performance reasons, the hybrid :meth:`nvidia.dali.ops.ImageDecoder` operator, which is
  nvJPEG based, creates threads on its own, and these threads are always affined.

In ``DALI_AFFINITY_MASK``, if the number of threads is higher than the number of CPU IDs,
the following process is applied:

1) The threads are assigned to the CPU IDs until all of the CPU IDs from ``DALI_AFFINITY_MASK``
   are assigned.
2) For the remaining threads, the CPU IDs from nvmlDeviceGetCpuAffinity will be used.

An example:

.. code-block:: bash

  num_threads=5
  DALI_AFFINITY_MASK=3,5,6,10

This example sets thread 0 to CPU 3, thread 1 to CPU 5, thread 2 to CPU 6, thread 3 to CPU 10,
and thread 4 to the CPU ID that is returned by nvmlDeviceGetCpuAffinity.

Memory Consumption
------------------

DALI uses the following memory types:

- Host
- Host-page-locked
- GPU

Allocating and freeing GPU and host page-locked (or pinned) memory require
device synchronization. As a result, when possible, DALI avoids reallocating these kinds of memory.
The buffers that are allocated with these storage types will only grow when the existing buffer is too
small to accommodate the requested shape. This strategy reduces the number of total memory
management operations and increases the processing speed when the memory requirements become stable
and no more allocations are required.

By contrast, ordinary host memory is relatively inexpensive to allocate and free. To reduce
host memory consumption, the buffers might shrink when the new requested size is smaller than
the fraction of the old size. This is called *shrink threshold*. It can be adjusted to a value
between 0 (never shrink) and 1 (always shrink). The default is 0.9. The value can be controlled
by the ``DALI_HOST_BUFFER_SHRINK_THRESHOLD`` environmental variable or be set in Python by
calling the `nvidia.dali.backend.SetHostBufferShrinkThreshold` function.

During processing, DALI works on batches of samples. For GPU and some CPU operators, each batch
is stored as contiguous memory and is processed at once, which reduces the number of
necessary allocations. For some CPU operators that cannot calculate their output size ahead of
time, the batch is stored as a vector of separately allocated samples.

For example, if your batch consists of nine 480p images and one 4K image in random order, the
contiguous allocation can accommodate all possible combinations of these batches. On the other
hand, the CPU batch that is stored as separate buffers needs to keep a 4K allocation for every
sample after several iterations. The GPU buffers that keep the operator outputs can grow are
as large as the largest possible batch, whereas the non-contiguous CPU buffers can reach
the size of the largest sample in the data set multiplied by the number of samples in the batch.

The host and the GPU buffers have a configurable growth factor. If the factor is greater than 1, and
to potentially avoid subsequent reallocations.
This functionality is disabled by default, and the growth factor is set to 1. The growth factors
can be controlled with the ``DALI_HOST_BUFFER_GROWTH_FACTOR`` and ``DALI_DEVICE_BUFFER_GROWTH_FACTOR``
environmental variables and with the `nvidia.dali.backend.SetHostBufferGrowthFactor` and
`nvidia.dali.backend.SetDeviceBufferGrowthFactor` Python API functions.
For convenience, the DALI_BUFFER_GROWTH_FACTOR environment variable and the
`nvidia.dali.backend.SetBufferGrowthFactor` Python function can be used to set the same
growth factor for the host and the GPU buffers.

Operator Buffer Presizing
-------------------------

When you can precisely forecast the memory consumption during a DALI run, this functionality helps
you fine tune the processing pipeline. One of the benefits is that the overhead of some
reallocations can be avoided.

DALI uses intermediate buffers to pass data between operators in the processing graph. The capacity
of this buffers is increased to accommodate new data, but is never reduced. Sometimes, however,
even this limited number of allocations might still affect DALI performance.
If you know how much memory each operator buffer requires, you can add a hint to preallocate the
buffers before the pipeline is first run.

The following parameters are available:

- The ``bytes_per_sample`` pipeline argument, which accepts one value that is used globally across
  all operators for all buffers.
- The ``bytes_per_sample_hint`` per operator argument, which accepts one value or a list of values.

When one value is provided, it is used for all output buffers for an operator. When a list is
provided, each operator output buffer is presized to the corresponding size.
To determine the amount of memory output that each operator needs, complete the following tasks:

1) Create the pipeline by setting ``enable_memory_stats`` to True.
2) Query the pipeline for the operator's output memory statistics by calling the
   :meth:`nvidia.dali.pipeline.Pipeline.executor_statistics` method on the pipeline.

The ``max_real_memory_size`` value represents the biggest tensor in the batch for the outputs that
allocate memory per sample and not for the entire batch at the time or the average tensor size when
the allocation is contiguous. This value should be provided to ``bytes_per_sample_hint``.

Prefetching Queue Depth
-----------------------

The DALI pipeline allows the buffering of one or more batches of data, which is important when
the processing time varies from batch to batch.
The default prefetch depth is 2. You can change this value by using the ``prefetch_queue_depth``
pipeline argument. If the variation is not hidden by the default prefetch depth value,
we recommend that you prefetch more data ahead of time.

.. note::
  Increasing queue depth also increases memory consumption.

Running DALI pipeline
---------------------

DALI pipeline can be run in one of the following ways:

- | Simple run method, which runs the computations and returns the results.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.BASIC` API type.
- | :meth:`nvidia.dali.pipeline.Pipeline.schedule_runs`,
    :meth:`nvidia.dali.pipeline.Pipeline.share_outputs`,
    :meth:`nvidia.dali.pipeline.Pipeline.release_outputs` that allows a fine-grain control for
    the duration of the output buffers' lifetime.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.SCHEDULED` API type.
- | Built-in iterators for MXNet, PyTorch, and TensorFlow.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.ITERATOR` API type.

The first API, :meth:`nvidia.dali.pipeline.Pipeline.run()` method completes the following tasks:

#. Launches the DALI pipeline.
#. Executes the prefetch iterations if necessary.
#. Waits until the first batch is ready.
#. Returns the resulting buffers.

Buffers are marked as in-use until the next call to
:meth:`nvidia.dali.pipeline.Pipeline.run`. This process can be wasteful because the data is usually
copied to the DL framework's native storage objects and DALI pipeline outputs could be returned to
DALI for reuse.

The second API, which consists of :meth:`nvidia.dali.pipeline.Pipeline.schedule_run()`,
:meth:`nvidia.dali.pipeline.Pipeline.share_outputs()`, and :meth:`nvidia.dali.pipeline.Pipeline.release_outputs()`
allows you to explicitly manage the lifetime of the output buffers. The
:meth:`nvidia.dali.pipeline.Pipeline.schedule_run()` method instructs DALI to prepare the next
batch of data, and, if necessary, to prefetch. If the execution mode is set to asynchronous,
this call returns immediately, without waiting for the results. This way, another task can be
simultaneously executed. The data batch can be requested from DALI by calling
:meth:`nvidia.dali.pipeline.Pipeline.share_outputs`, which returns the result buffer. If the data
batch is not yet ready, DALI will wait for it. The data is ready as soon as the
:meth:`nvidia.dali.pipeline.Pipeline.share_outputs()`` is complete. When the DALI buffers are
no longer needed, because data was copied or has already been consumed, call
:meth:`nvidia.dali.pipeline.Pipeline.release_outputs()` to return the DALI buffers for reuse
in subsequent iterations.

Built-in iterators use the second API to provide convenient wrappers for immediate use in
Deep Learning Frameworks. The data is returned in the framework's native buffers. The iterator's
implementation copies the data internally from DALI buffers and recycles the data by calling
:meth:`nvidia.dali.pipeline.Pipeline.release_outputs()`.

We recommend that you do not mix the APIs. The APIs follow a different logic for the output
buffer lifetime management, and the details of the process are subject to change without notice.
Mixing the APIs might result in undefined behavior, such as a deadlock or an attempt to access
an invalid buffer.

Sharding
--------

Sharding allows DALI to partition the dataset into nonoverlapping pieces on which each DALI pipeline
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

DALI is used in the Deep Learning Frameworks through dedicated iterators, and these iterators need
to be aware of this padding and other reader properties.

Here are the iterator options:

- ``fill_last_batch`` – Determines whether the last batch should be full, regardless of whether
  the shard size is divisible by the batch size.
- | ``reader_name`` - Allows you to provide the name of the reader that drives the iterator and
   provides the necessary parameters.

  .. note::
    We recommend that you use this option. With this option, the next two options are excluded and
    cannot be used.

  | This option is more flexible and accurate and takes into account that shard size for a pipeline
    can differ between epochs when the shards are rotated.
- ``size``: Displays the size of the shard for an iterator or, if there is more than one shard,
  the sum of all shard sizes for all wrapped pipelines.
- | ``last_batch_padded``: Determines whether the tail of the data consists of data from the next
    shard (``False``) or is duplicated dummy data (``True``).
  | It is applicable when the shard size is not a multiple of the batch size,
- | ``last_batch_policy`` - Determines the police about the last batch when the shard size is not
    divisible by the batch size.
  | Only partially filled with the data or dropped entirely if it. The possible options are:

  - | FILL - Fills the last batch with the data wrapping up the data set.
    | The precise behavior depends on the reader which may duplicate the last sample to fill
      the batch.
  - DROP - If the last batch cannot be fully filled by the data from given epoch it is dropped.
  - PARTIAL - Returns the part of the last batch filled with the data relevant to given epoch.

Here is the formula to calculate the shard size for a shard ID:

*floor((id + 1) * dataset_size / num_shards) - floor(id * dataset_size / num_shards)*

When the pipeline advances through the epochs and the reader moves to the next shard, the formula
needs to be extended to reflect this change:

*floor(((id + epoch_num) % num_shards + 1) * dataset_size / num_shards) - floor(((id + epoch_num) % num_shards) * dataset_size / num_shards)*

When the second formula is used, providing a size value once at the beginning of the training works
only when the ``stick_to_shard`` reader option is enabled and prevents DALI from rotating shards.
When this occurs, use the first formula.

To address these challenges, use the ``reader_name`` parameter and allow the iterator
handle the details.

C++ API
-------

.. note::
  **This feature is not officially supported and may change without notice**

The C++ API allows you to use DALI as a library from native applications. Refer to
the ``PipelineTest`` family of tests for more information about how to use this API.
