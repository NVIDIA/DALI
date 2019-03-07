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
The number of DALI threads are set during the pipeline construction by num_threads argument, and the set_affinity enables setting the thread affinity for the CPU worker threads. Note that for performance reasons the nvJPEG decoder operator houses threads on its own, and they are always affined.
If the number of threads are more than the CPU id numbers in the environment variable ``DALI_AFFINITY_MASK``, then the following applies: First, the threads are assigned to CPU id numbers until all CPU id numbers from ``DALI_AFFINITY_MASK are assigned``. Then, for the remaining threads, the CPU id numbers from nvmlDeviceGetCpuAffinity will be used. See below example:

.. code-block:: bash

  num_threads=5
  DALI_AFFINITY_MASK=3,5,6,10

this will set thread 0 to CPU 3, thread 1 to CPU 5, thread 2 to CPU 6, thread 3 to CPU 10 and thread 4 to CPU id that is returned by nvmlDeviceGetCpuAffinity.


Memory consumption
------------------

In DALI architecture all GPU operators process the whole batch at a time while the CPU operator processes one sample at a time.
With DALI the memory is only growing, never shrinking. Hence each GPU buffer is as large as the largest possible batch, while the CPU buffers are as large as batch size multiplied by the size of the largest sample. Note that even though the CPU processes one sample at a time per thread, a vector of samples need to reside in the memory.
This lazy allocation strategy reduces the number of total memory operations, and hence helps in cases when the memory allocation is expensive, like the pinned CPU memory or GPU memory.
This is most visible for the operators whose output size may differ from sample to sample and from run to run. Operator with the fixed size outputs, such as crop, does not influence the overall memory consumption growth over time

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

C++ API
-------

.. note::

  **This feature is not officially supported and may change without notice**

The C++ API enables using DALI as a library from native applications. **The API is experimental, unstable and can change without notice**. Refer to ``PipelineTest`` family of tests for how to use the C++ API.
