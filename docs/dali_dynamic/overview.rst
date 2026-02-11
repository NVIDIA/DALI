.. _dynamic_api_overview:
Dynamic API Overview
====================

.. warning::
   DALI Dynamic is an experimental feature and is subject to change.

**DALI Dynamic** extends NVIDIA DALI by introducing an imperative execution model with lazy evaluation.
It complements the existing graph-based pipeline execution, and its main goal is to enable seamless
integration into Python workflows. This results in easier debugging and opens a path to rapid
prototyping of pre-processing pipelines.

The imperative style of DALI Dynamic provides several advantages:

- Operator execution can be controlled directly within iterative workflows.
- Integration with existing Python code and libraries is simplified.
- Pipelines can be written and debugged incrementally.

You can find more details in the
`Getting Started with Dynamic Mode tutorial <../examples/getting_started/dynamic_mode.ipynb>`_.

The **DALI Dynamic API** is available in the :mod:`nvidia.dali.experimental.dynamic` module,
typically imported and referred to as ``ndd``.
You can also access the :ref:`DALI Pipeline API <pipeline_api_overview>`, which is DALI's long-standing
graph-based execution model.


How it works
------------

Dynamic Mode
^^^^^^^^^^^^

DALI Dynamic offers an imperative programming model. This means that it does not
require isolating data processing logic into a separate pipeline, and operators can
be called directly from Python code. Moreover, DALI Dynamic evaluates operators asynchronously
to improve performance.

.. code-block:: python

   import nvidia.dali.experimental.dynamic as ndd

   model = MyModel(...)
   flip_horizontal = True
   flip_vertical = False
   dataset = ndd.readers.File(file_root=images_dir)
   for jpegs, labels in dataset.next_epoch(batch_size=16):
       img = ndd.decoders.image(jpegs, device="gpu")
       flipped = ndd.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
       model((flipped, img))

In the above example, batches of images are read from a dataset, then decoded and processed using
DALI operators to create a model input. All of this can be integrated directly into an existing code
flow.


Key features
^^^^^^^^^^^^

- **Imperative programming with asynchronous or lazy execution**
  In the DALI Pipeline API, operators were defined and executed within a static
  pipeline graph. This often resulted in a steep learning curve, complex debugging, and
  limited error visibility. DALI Dynamic introduces imperative programming aligning DALI more closely
  with standard Python workflows. DALI Dynamic executes the operators asynchronously in the background
  to transparently improve performance, but can be configured to run fully lazily or synchronously
  with the main thread.

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
   can be performed in DALI Dynamic using exactly the same operators as in the Pipeline API
   and transition between the two modes is straightforward.
