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

  # assuming that DALI uses num_threads=5
  DALI_AFFINITY_MASK="3,5,6,10"

This example sets thread 0 to CPU 3, thread 1 to CPU 5, thread 2 to CPU 6, thread 3 to CPU 10,
and thread 4 to the CPU ID that is returned by nvmlDeviceGetCpuAffinity.

Memory Consumption
------------------

.. currentmodule:: nvidia.dali.backend

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


Allocator Configuration
-----------------------

DALI uses several types of memory resources for various kinds of memory.

For regular host memory, DALI uses (aligned) ``malloc`` for small allocations and a custom memory
pool for large allocations (to prevent costly calls to ``mmap`` by ``malloc``).
The maximum size of a host allocation that is allocated directly with ``malloc`` can be customized
by setting ``DALI_MALLOC_POOL_THRESHOLD`` environment variable. If not specified, the value is
either derived from environment variables controlling ``malloc`` or, if not found, a default value
of 32M is used.

For host pinned memory, DALI uses a stream-aware memory pool on top of ``cudaMallocHost``.
Direct usage of ``cudaMallocHost``, while discouraged, can be forced by specifying
``DALI_USE_PINNED_MEM_POOL=0`` in the environment.

For device memory, DALI uses a stream-aware memory pool built on top of CUDA VMM resource
(if the platform supports VMM). It can be changed to ``cudaMallocAsync`` or even plain
``cudaMalloc``.
In order to skip memory pool entirely and use ``cudaMalloc`` (not recommended), set
``DALI_USE_DEVICE_MEM_POOL=0``.
Set ``DALI_USE_CUDA_MALLOC_ASYNC=1`` to use ``cudaMallocAsync`` instead of DALI's internal memory
pool.
When using the memory pool (``DALI_USE_DEVICE_MEM_POOL=1`` or unset), you can disable the use of
VMM by setting ``DALI_USE_VMM=0``. This will cause ``cudaMalloc`` to be used as an upstream memory
resource for the internal memory pool.

Using ``cudaMallocAsync`` typically results in slightly slower execution, but it enables memory
pool sharing with other libraries using the same allocation method.

.. warning::
    Disabling memory pools will result in a dramatic drop in performance. This option is provided
    only for debugging purposes.

    Disabling CUDA VMM can degrade performance due to pessimistic synchronization in
    ``cudaFree``, and it can cause out-of-memory errors due to fragmentation affecting
    ``cudaMalloc``.


Memory Pool Preallocation
-------------------------

DALI uses several memory pools - one for each CUDA device plus one global pool for pinned host
memory. Normally these pools grow on demand. The growth can result in temporary drop in throughput.
In performance-critical applications, this can be avoided by preallocating the pools.

.. autofunction:: PreallocateDeviceMemory
.. autofunction:: PreallocatePinnedMemory


Freeing Memory Pools
--------------------

The memory pools used by DALI are global and shared between pipelines. The destruction
of a pipeline doesn't cause the memory to be returned to the operating system - it's kept
for future pipelines. To release memory that's not currently in use, you can call
:func:`nvidia.dali.backend.ReleaseUnusedMemory`

.. autofunction:: ReleaseUnusedMemory


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

Readers fine-tuning
-------------------

Selected readers use environment variables to fine-tune their behavior regarding the file storage
access patterns. In most cases the default parameter should provide decent performance, still in
some particular system properties (file system type) can require adjusting them accordingly.

- ``DALI_GDS_CHUNK_SIZE`` - adjust the size of a single GDS read request.

  Applicable to the :meth:`nvidia.dali.fn.readers.numpy` operator for the ``GPU`` backend.
  The default value is 2MB. It must be a number, optionally followed by 'k' or 'M',
  be a power of two, and not be larger than 16MB. The optimal performance can be achieved for
  different values depending on the filesystem and GDS version.

- ``DALI_ODIRECT_ALIGNMENT`` - adjusts the O_DIRECT alignment.

  Applicable only to readers that expose `use_o_direct` parameter, like
  :meth:`nvidia.dali.fn.readers.numpy` operator for the ``CPU`` backend. The default value is 4KB.
  It must be a number, optionally followed by 'k' or 'M', be a power of two, and not be larger
  than 16MB. The minimal value depends on the file system. See more in
  `the Linux Open call manpage, O_DIRECT section <https://man7.org/linux/man-pages/man2/open.2.html>`__

- ``DALI_ODIRECT_LEN_ALIGNMENT`` - adjusts the O_DIRECT read length alignment.

  Applicable only to readers that expose `use_o_direct` parameter, like
  :meth:`nvidia.dali.fn.readers.numpy` operator for the ``CPU`` backend. The default value is 4KB.
  It must be a number, optionally followed by 'k' or 'M', be a power of two, and not be larger
  than 16MB. The minimal value depends on the file system. See more in
  `the Linux Open call manpage, O_DIRECT section <https://man7.org/linux/man-pages/man2/open.2.html>`__

- ``DALI_ODIRECT_CHUNK_SIZE`` - adjust the size of single O_DIRECT read request.

  Applicable only to readers that expose `use_o_direct` parameter, like
  :meth:`nvidia.dali.fn.readers.numpy` operator for the ``CPU`` backend. The default value is 2MB.
  It must be a number, optionally followed by 'k' or 'M', be a power of two, and not larger
  than 16MB, and not smaller than ``DALI_ODIRECT_LEN_ALIGNMENT``. The optimal performance can be
  achieved for different values depending on the filesystem.

