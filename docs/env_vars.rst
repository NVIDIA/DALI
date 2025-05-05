DALI Environment Variables
==========================

This page lists environment variables used by DALI.
Note that those variables often control internal behavior of DALI and are subject to change
without notice.

Memory management
~~~~~~~~~~~~~~~~~

`DALI_USE_DEVICE_MEM_POOL`
--------------------------

Values: 0, 1

Default: 1

If nonzero, DALI uses a memory pool for device memory; otherwise plain ``cudaMalloc`` is used.

.. warning::
    Specify 0 for debugging only, doing so will drastically degrade performace.

`DALI_USE_VMM`
--------------

Values: 0, 1

Default: 1

If nonzero, DALI uses a memory pool based on virtual memory management functions
(``cuMemCreate``, ``cuMemMap``, etc) when possible. If disabled, DALI will use a pool based on
`cudaMalloc`.

`DALI_USE_PINNED_MEM_POOL`
--------------------------

Values: 0, 1

Default: 1

If nonzero, DALI uses a memory pool for device-accessible host memory; otherwise plain
``cudaMallocHost`` is used.

.. warning::
    Specify 0 for debugging only, doing so will drastically degrade performace.

`DALI_USE_CUDA_MALLOC_ASYNC`
----------------------------

Values: 0, 1

Default: 0

If enabled, DALI will use ``cudaMallocAsync`` as the device memory allocation mechanism.
This flag is not compatible with ``DALI_USE_DEVICE_MEM_POOL=0`` and explicitly enabling
``DALI_USE_VMM=1``.

`DALI_MALLOC_POOL_THRESHOLD`
----------------------------

Values: >= 0, optionally followed by k for KiB or M for MiB.

Default: 0

If nonzero, dali uses an internal memory pool for regular host memory below the specified size.

`DALI_GDS_CHUNK_SIZE`
---------------------

Values: powers of 2, 4k to 16M, with optional k or M suffix

Default: 2M

The size of memory chunk used by GPU Direct Storage in DALI.

`DALI_HOST_BUFFER_SHRINK_THRESHOLD`
-----------------------------------

Values: floating point, 0..1

Default: 0.5

If set, buffers resized to below the given fraction of capacity will be shrunk to fit the data.

`DALI_BUFFER_GROWTH_FACTOR`
---------------------------

Values: floating point, >=1

Default: 1.1

When greather than 1, buffers are allocated with allowance to avoid frequent reallocation.

`DALI_HOST_BUFFER_GROWTH_FACTOR`
--------------------------------

Specialized version of `DALI_BUFFER_GROWTH_FACTOR` for CPU buffers only.

`DALI_DEVICE_BUFFER_GROWTH_FACTOR`
----------------------------------

Specialized version of `DALI_BUFFER_GROWTH_FACTOR` for GPU buffers only.

`DALI_RESTRICT_PINNED_MEM`
--------------------------

Values: 0, 1

Default: 0

If enabled, DALI will reduce the use of pinned host memory; useful on systems where the amount
of pinned memory is restricted (e.g. WSL).

Image decoding
~~~~~~~~~~~~~~

`DALI_MAX_JPEG_SCANS`
---------------------

Values: >= 1

Default: 256

The maximum number of progressive JPEG scans. Specify lower values to mitigate maliciously malformed
JPEGs, designed for denial of service attacks.

`DALI_NVIMGCODEC_LOG_LEVEL`
---------------------------

Values: 1..5

Default: 2

The verbosity of logs produced by nvJPEG

Miscellaneous
~~~~~~~~~~~~~

`DALI_OPTIMIZE_GRAPH`
---------------------

Values: 0, 1

Default: 1

For debugging only; if set to 0, all DALI graph optimizations are disabled.

`DALI_ENABLE_CSE`
-----------------

Values: 0, 1

Default: 1

For debugging only; if set to 0, the common subexpression elimination (CSE) graph optimization
is disabled. If `DALI_OPTIMIZE_GRAPH` is disabled, this flag has no effect.


`DALI_USE_EXEC2`
----------------

Values: 0, 1

Default: 0

If set, DALI will use the dynamic executor (as if ``exec_dynamic=True`` was set in the  Pipeline)
whenever the default asychronous pipelined execution with uniform queue size is specified.
Enabling the dynamic executor can potentially improve memory consumption.

.. note::
    This flag is used in the backend only; the Python frontend is unaware of it and doesn't enable
    the new features and optimizations that would be enabled by specifying ``exec_dynamic=True``
    in the pipeline.

`DALI_AFFINITY_MASK`
--------------------

Values: comma-separated list of CPU IDs

Default: empty

Sets the thread affinity. The number of entries in the list must match the ``num_threads`` passed
to the pipeline.
Requires NVML.


`DALI_PRELOAD_PLUGINS`
----------------------

Values: colon-separated list of paths

Default: empty

If specified, DALI will preload plugins specified in this list.

`DALI_DISABLE_NVML`
-------------------

Values: 0, 1

Default: 0

If set, DALI doesn't try to use NVML. Useful on systems without NVML support, e.g. WSL2.

Network
~~~~~~~

`DALI_S3_NO_VERIFY_SSL`
-----------------------

Values: 0, 1

Default: 0

If set, DALI will not verify SSL certificates when communicating with S3 services.

By default, DALI uses SSL when communicating with S3 services, which includes verifying SSL certificates.
This option allows you to override the default behavior of verifying SSL certificates.

Testing
~~~~~~~

`DALI_EXTRA_PATH`
-----------------

Values: path

Default: empty

The path to where the contents of `DALI_extra <https://github.com/NVIDIA/DALI_extra>`_ repository
reside. Necessary to run DALI tests.
