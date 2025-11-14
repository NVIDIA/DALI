Overview
========

**DALI Dynamic** extends NVIDIA DALI by introducing an imperative execution model with lazy evaluation.
It complements the existing graph-based pipeline execution, and its main goal is to enable seamless
integration into Python workflows. This results in easier debugging and opens a path to rapid 
prototyping of pre-processing pipelines.

Key features include:

- **Imperative programming with lazy execution**  
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
   an alternative interface for a seamless Python experience. Prototyping and development 
   can be performed in DALI Dynamic using exactly the same operators as a pipeline mode
   and transition between the two modes is straightforward.


How it works
------------

Dynamic Mode
^^^^^^^^^^^^

DALI Dynamic offers an imperative programming model. This means that it does not 
require isolating data processing logic into a separate pipeline, and operators can
be called directly from Python code. Moreover, DALI Dynamic evaluates operators lazily 
to improve performance. 

.. code-block:: python

   import nvidia.dali.experimental.dynamic as ndd

   model = MyModel(...)
   flip_horizontal = True
   flip_vertical = False
   dataset = ndd.readers.file(file_root=images_dir)
   for batch in dataset.epoch(batch_size=16):
       img = ndd.decoders.image(batch, device="mixed")
       flipped = ndd.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
       model((flipped, img))

In the above example, batches of images are read from a dataset, then decoded and processed using 
DALI operators to create a model input. All of this is integrated directly into an existing code 
flow.

This imperative style provides several advantages:

- Operator execution can be controlled directly within iterative workflows.
- Integration with existing Python code and libraries is simplified.
- Pipelines can be written and debugged incrementally.

Pipeline Mode
^^^^^^^^^^^^^

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

This feature is still available and can have a slightly better performance than DALI Dynamic.
Please refer to :ref:`pipeline<Pipeline>` for details.