Performance Tuning
==================

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
  For performance reasons, the hybrid :meth:`nvidia.dali.fn.decoders.image` operator, which is
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
   :meth:`nvidia.dali.Pipeline.executor_statistics` method on the pipeline.

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
