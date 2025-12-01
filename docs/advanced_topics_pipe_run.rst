Pipeline Run Methods
====================

DALI pipeline can be run in one of the following ways:

- | Simple run method, which runs the computations and returns the results.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.BASIC` API type.
- | :meth:`nvidia.dali.Pipeline.schedule_run`,
    :meth:`nvidia.dali.Pipeline.share_outputs`,
    :meth:`nvidia.dali.Pipeline.release_outputs` that allows a fine-grain control for
    the duration of the output buffers' lifetime.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.SCHEDULED` API type.
- | Built-in iterators for PyTorch, JAX, PaddlePaddle, and TensorFlow.
  | This option corresponds to the :meth:`nvidia.dali.types.PipelineAPIType.ITERATOR` API type.

The first API, :meth:`nvidia.dali.Pipeline.run` method completes the following tasks:

#. Launches the DALI pipeline.
#. Executes the prefetch iterations if necessary.
#. Waits until the first batch is ready.
#. Returns the resulting buffers.

Buffers are marked as in-use until the next call to
:meth:`nvidia.dali.Pipeline.run`. This process can be wasteful because the data is usually
copied to the DL framework's native storage objects and DALI pipeline outputs could be returned to
DALI for reuse.

The second API, which consists of :meth:`nvidia.dali.Pipeline.schedule_run`,
:meth:`nvidia.dali.Pipeline.share_outputs`, and :meth:`nvidia.dali.Pipeline.release_outputs`
allows you to explicitly manage the lifetime of the output buffers. The
:meth:`nvidia.dali.Pipeline.schedule_run` method instructs DALI to prepare the next
batch of data, and, if necessary, to prefetch. If the execution mode is set to asynchronous,
this call returns immediately, without waiting for the results. This way, another task can be
simultaneously executed. The data batch can be requested from DALI by calling
:meth:`nvidia.dali.Pipeline.share_outputs`, which returns the result buffer. If the data
batch is not yet ready, DALI will wait for it. The data is ready as soon as the
:meth:`nvidia.dali.Pipeline.share_outputs`` is complete. When the DALI buffers are
no longer needed, because data was copied or has already been consumed, call
:meth:`nvidia.dali.Pipeline.release_outputs` to return the DALI buffers for reuse
in subsequent iterations.

Built-in iterators use the second API to provide convenient wrappers for immediate use in
Deep Learning Frameworks. The data is returned in the framework's native buffers. The iterator's
implementation copies the data internally from DALI buffers and recycles the data by calling
:meth:`nvidia.dali.Pipeline.release_outputs`.

We recommend that you do not mix the APIs. The APIs follow a different logic for the output
buffer lifetime management, and the details of the process are subject to change without notice.
Mixing the APIs might result in undefined behavior, such as a deadlock or an attempt to access
an invalid buffer.
