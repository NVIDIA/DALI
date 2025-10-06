DALI Dynamic
========

Overview
--------

**DALI Dynamic** extends NVIDIA DALI by introducing an imperative execution model with lazy evaluation.  
It complements the existing graph-based pipeline execution and is intended for rapid prototyping, 
easier debugging, and seamless integration into existing Python workflows.

Key features include:

- **Lazy execution**  
  In the original DALI execution model, operators were defined and executed within a static 
  pipeline graph. This often resulted in a steep learning curve, complex debugging, and 
  limited error visibility. DALI Dynamic introduces imperative programming with lazy operator 
  execution, aligning DALI more closely with standard Python workflows.  

- **Minimal performance overhead**  
  DALI Dynamic is designed to deliver performance that is close to graph-based pipelines, incurring 
  only marginal overhead.  

- **Batch processing support**  
  Batch processing remains a core concept in DALI. DALI Dynamic preserves this functionality and 
  introduces a dedicated API for batch-oriented workflows.  

- **Framework interoperability**  
  DALI Dynamic provides type conversion support for major deep learning frameworks, including 
  PyTorch, CuPy, JAX, and Numba.

.. note::

   DALI Dynamic does not replace the graph-based execution model. Instead, it provides 
   an alternative interface for defined use cases. Prototyping and development 
   can be performed in DALI Dynamic, with subsequent transition to pipelines for 
   high-performance, production scenarios.


How it works
------------

DALI traditionally relies on explicitly defined pipelines, where data processing 
is specified as a computation graph. A typical usage pattern involves defining 
a decorated function with ``@pipeline_def``, building the pipeline, and executing 
it with batched data:

.. code-block:: python

   @pipeline_def
   def my_pipe(flip_vertical, flip_horizontal):
       """Creates a DALI pipeline that returns flipped and original images."""
       data, _ = fn.readers.file(file_root=images_dir)
       img = fn.decoders.image(data, device="mixed")
       flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
       return flipped, img

   pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)
   flipped, img = pipe.run()

In contrast, DALI Dynamic enables imperative programming. Operators are invoked directly 
from Python without building an explicit graph, while still leveraging lazy evaluation 
for efficiency:

.. code-block:: python

   import nvidia.dali.dynamic as ndd

   dataset = ndd.readers.file(file_root=images_dir)
   for batch in dataset.epoch(batch_size=16):
       img = ndd.decoders.image(batch, device="mixed")
       flipped = ndd.flip(img, horizontal=True, vertical=False)

This imperative style provides several advantages:

- Pipelines can be written and debugged incrementally.
- Integration with existing Python code and libraries is simplified.
- Operator execution can be controlled directly within iterative workflows.


Summary
-------

DALI Dynamic introduces an imperative, Python-native programming model to NVIDIA DALI, 
while retaining the benefits of lazy execution and batch processing.  

Compared to graph-based pipelines, it offers greater flexibility and improved 
debuggability, with only a minor performance trade-off.  

For high-performance or production use cases, users are encouraged to transition 
from DALI Dynamic to traditional graph-based execution.
