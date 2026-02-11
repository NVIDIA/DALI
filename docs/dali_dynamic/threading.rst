Multithreading
==============

.. currentmodule:: nvidia.dali.experimental.dynamic

Thread Safety
-------------

Dynamic mode generally is thread-safe and supports
`free-threaded Python <https://docs.python.org/3/howto/free-threading-python.html>`__. Operators can
be called from multiple threads concurrently, and :class:`Tensor` and :class:`Batch` objects can be
safely passed between threads.

The one limitation is that a single :class:`EvalContext` instance must not be active in multiple
threads simultaneously. Because the default evaluation context is thread-local (each thread
automatically gets its own), this is only an issue when it is manually created and shared across
threads.

:octicon:`alert-fill;1.2em;align-text-bottom text-warning` Multiple threads using the same :class:`EvalContext`:

.. code-block:: python

   import threading
   import nvidia.dali.experimental.dynamic as ndd

   ctx = ndd.EvalContext(num_threads=4)

   def worker():
       with ctx:  # Bad: using the same EvalContext in multiple threads simultaneously
           img = ndd.random.uniform(shape=(100, 100, 3), range=(0, 255), dtype=ndd.uint8)
           flipped = ndd.flip(img, horizontal=True)
           ...

   threads = [threading.Thread(target=worker) for _ in range(4)]
   for t in threads:
       t.start()
   for t in threads:
       t.join()

Here, the code should either create an instance of the evaluation context per thread, or use
:func:`set_num_threads`.

.. warning::

   :func:`set_num_threads` controls DALI's internal thread pool. It is unrelated to Python-level
   multithreading.

Thread-local storage
--------------------

The context managers :class:`EvalMode`, :class:`EvalContext`, and :class:`Device` all use
thread-local stacks, allowing each thread to independently choose its eval mode, execution context,
and device without affecting other threads.

CUDA stream configuration can also be thread-local. The function :func:`set_current_stream`
sets the stream for the calling thread only, while :func:`set_default_stream` is global.
Technically, :func:`set_current_stream` sets the stream of the current thread's default
evaluation context but this is equivalent for most practical purposes.

.. code-block:: python

   import threading
   import nvidia.dali.experimental.dynamic as ndd

   def worker():
       with ndd.EvalMode.deferred:
           img = ndd.random.uniform(shape=(100, 100, 3), range=(0, 255), dtype=ndd.uint8)
           ...

   # The main thread's EvalMode is unaffected by the worker thread
   t = threading.Thread(target=worker)
   t.start()
   t.join()

.. tip::

   When using :attr:`EvalMode.deferred`, be cautious about sharing tensors between threads. You
   might want to explicitly call :func:`~Tensor.evaluate` before sending data from a worker thread
   to the main thread.
