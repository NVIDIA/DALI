Advanced topics
=================

This section covers a few advanced topics that are mentioned in the API description. However, for the typical use case the default DALI configuration will perform well out of the box, and you don't need to worry about these advanced topics.

DALI internal data format
-------------------------

Internally, DALI processes data in NHWC format. To reshape it to NCHW, please use ``\*Permute`` operator family. Also note that if such operators are used in the middle of a pipeline, then the subsequent operations in the pipeline will make incorrect assumptions about the processed data format, and the result will be undefined.

Thread affinity
---------------

The purpose of this functionality is to allow pinning DALI threads to the given CPU. Thread affinity can save the overhead of worker threads jumping from core to core. Hence this functionality can increase the performance for CPU heavy workloads.
It is possible to set the CPU thread affinity for DALI using the ```DALI_AFFINITY_MASK``` environment variable. ```DALI_AFFINITY_MASK`` should be a comma separated list of CPU ids numbers which will be assigned to corresponding DALI threads.
The number of DALI threads are set during the pipeline construction by num_threads argument, and the set_affinity enables setting the thread affinity for the CPU worker threads. Note that for performance reasons the hybrid ```ImageDecoder``` operator (nvJPEG based) houses threads on its own, and they are always affined.
If the number of threads are more than the CPU id numbers in the environment variable ``DALI_AFFINITY_MASK``, then the following applies: First, the threads are assigned to CPU id numbers until all CPU id numbers from ``DALI_AFFINITY_MASK are assigned``. Then, for the remaining threads, the CPU id numbers from nvmlDeviceGetCpuAffinity will be used. See below example:

.. code-block:: bash

  num_threads=5
  DALI_AFFINITY_MASK=3,5,6,10

this will set thread 0 to CPU 3, thread 1 to CPU 5, thread 2 to CPU 6, thread 3 to CPU 10 and thread 4 to CPU id that is returned by nvmlDeviceGetCpuAffinity.


Memory consumption
------------------

DALI uses three kinds of memory: host, host page-locked (pinned) and GPU.
GPU and host pinned memory allocation and freeing require device synchronization. For this reason, DALI avoids reallocating these kinds of memory whenever possible. The buffers allocated with this kind of storage will only grow when the existing buffer is too small to accommodate the requested shape. This allocation strategy reduces the number of total memory management operations and greatly increases the processing speed after the allocations have stabilized.

In contrast, ordinary host memory is relatively cheap to allocate and free. To reduce host memory consumption, the buffers may shrink if the new requested size is smaller than a specified fraction of the old size (called shrink threshold). It can be adjusted to any value between 0 (never shrink) and 1 (always shrink), with the default being 0.9. The value can be controlled either via environment variable `DALI_HOST_BUFFER_SHRINK_THRESHOLD` or set in Python using `nvidia.dali.backend.SetHostBufferShrinkThreshold` function.

During processing, DALI works on batches of samples. For GPU and some CPU Operators, the batch is stored as continuous allocation which is processed in one go, which reduces the number of necessary allocations.
For some CPU Operators, that cannot calculate their output size ahead of time, the batch is instead stored as a vector of separately allocated samples.

For example, if your batch consists of nine 480p images and one 4K image in random order, the single continuous allocation would be able to accommodate all possible combinations of such batches.
On the other hand, the CPU batch presented as separate buffers will need to keep a 4K allocation for every sample after several iterations.
In effect, GPU buffers are allocated to house transformation results is as large as the largest possible batch, while the CPU buffers can be as large as batch size multiplied by the size of the largest sample. Note that even though the CPU processes one sample at a time per thread, a vector of samples needs to reside in the memory.

Moreover, both host and GPU buffers have configurable growth factor - if it's above 1 and the requested new size exceeds buffer capacity, the buffer will be allocated with extra margin to potentially avoid subsequent reallocations. This functionality is disabled by default (the growth factor is set to 1). These factors can be controlled via environment variables `DALI_HOST_BUFFER_GROWTH_FACTOR` and `DALI_DEVICE_BUFFER_GROWTH_FACTOR`, respectively as well as with Python API functions `nvidia.dali.backend.SetHostBufferGrowthFactor` and `nvidia.dali.backend.SetDeviceBufferGrowthFactor`. For convenience, the variable `DALI_BUFFER_GROWTH_FACTOR` and corresponding Python function `nvidia.dali.backend.SetBufferGrowthFactor` set the same growth factor for host and GPU buffers.

Operator buffer presizing
-------------------------

The purpose of this functionality is to enable the user to fine-tune the processing pipeline in situations when it is possible to forecast precisely the memory consumption during a DALI run. This results in saving the overhead of some reallocations.
DALI uses intermediate buffers to pass data between operators in the processing graph. With DALI, the memory is never freed but just enlarged when present buffers are not sufficient to hold the data. However, in some cases, even this limited number of allocations still could affect DALI performance. Hence, if the user knows how much memory each operator buffer needs, then it is possible to provide a hint to presize buffers before the first run.
Two parameters are available: First, the ``bytes_per_sample`` pipeline argument, which accepts one value that is used globally across all operators for all buffers.
The second parameter is the ``bytes_per_sample_hint`` per operator argument, which accepts one value or a list of values. When one value is provided it is used for all output buffers for a given operator. When a list is provided then each buffer is presized to the corresponding size.

Prefetching queue depth
-----------------------

The purpose of this functionality is to average the processing time between batches when the variation from batch to batch is high.
DALI pipeline allows buffering one or more batches of data ahead. This becomes important when the data processing time between batches could vary a lot. Default prefetch depth is 2. The user can change this value using the ``prefetch_queue_depth`` pipeline argument. For example, if the variation is bigger then it is recommended to prefetch more ahead.

Running DALI pipeline
---------------------

DALI provides a couple of ways to run a pipeline:

- simple `run` method, which runs the computations and returns the results (corresponds to :meth:`nvidia.dali.types.PipelineAPIType.BASIC` API type)
- `schedule_run`, `share_outputs` and `release_outputs` with fine grain control of the output buffers' lifetime (corresponds to :meth:`nvidia.dali.types.PipelineAPIType.SCHEDULED` API type)
- built-in iterators for MXNet, PyTorch, and TensorFlow (corresponds to :meth:`nvidia.dali.types.PipelineAPIType.ITERATOR` API type)

The first API - :meth:`nvidia.dali.pipeline.Pipeline.run` method launches the DALI pipeline, executing prefetch iterations if necessary, waits until the first batch is ready and returns the resulting buffers. Buffers are marked as in-use untill the next call to `nvidia.dali.pipeline.Pipeline.run`. In many cases, it is wasteful as data is usually copied out to the native framework tensors after which they can be returned to DALI to be reused

The second API, consisting of :meth:`nvidia.dali.pipeline.Pipeline.schedule_run`, :meth:`nvidia.dali.pipeline.Pipeline.share_outputs` and :meth:`nvidia.dali.pipeline.Pipeline.release_outputs` allows the user to explicitly manage the lifetime of the output buffers. The :meth:`nvidia.dali.pipeline.Pipeline.schedule_run` method instructs DALI to prepare the next batch of data, prefetching if necessary. If the execution mode is set to asynchronous, this call returns immediately, without waiting for the results, so another task can be executed in parallel. The data batch can be requested from DALI by calling `share_outputs`, which returns the result buffer. If it is not ready yet, DALI will wait for it. The data is ready as soon as the :meth:`nvidia.dali.pipeline.Pipeline.share_outputs` method returns. When DALI buffers are no longer needed, because data was copied or already consumed, :meth:`nvidia.dali.pipeline.Pipeline.release_outputs` should be called to return DALI buffers to be reused in subsequent iterations.

Built-in iterators use the second API to provide convenient wrappers for immediate use in DL frameworks. The data is returned in framework's native buffers - the iterator's implementation internally copies the data from DALI buffers and recycles them by calling :meth:`nvidia.dali.pipeline.Pipeline.release_outputs`.

It is not recommended to mix any of the aforementioned APIs together, because they follow different logic for output buffer lifetime management and the details of the process are subject to change without notice. Mixing the APIs may result in an undefined behavior, like a deadlock or attempt to access an invalid buffer.


C++ API
-------

.. note::

  **This feature is not officially supported and may change without notice**

The C++ API enables using DALI as a library from native applications. **The API is experimental, unstable and can change without notice**. Refer to ``PipelineTest`` family of tests for how to use the C++ API.
