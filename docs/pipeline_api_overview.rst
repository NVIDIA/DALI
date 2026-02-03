.. _pipeline_api_overview:

Pipeline API Overview
=====================

DALI traditionally relies on explicitly defined pipelines, where data processing
is specified as a computation graph. A typical usage pattern involves defining
a decorated function with :ref:`@pipeline_def <pipeline_decorator>`,
constructing the :ref:`Pipeline object<pipeline_class>`, and executing it with batched data:

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


The :ref:`Pipeline class <pipeline_class>`, with the associated :ref:`data types <pipeline_types>` and
operations in :ref:`nvidia.dali.fn <operation reference>` and :ref:`nvidia.dali.math <mathematical expressions>`
modules are collectively called the **DALI Pipeline API** to distinguish it from the **DALI Dynamic API**.

:ref:`DALI Dynamic API <dali_dynamic>` allows you to run its operators in an imperative manner without
the necessity of defining the data processing graph upfront, at the cost of small performance overhead.


You can find more details in the
`Getting Started with Pipeline Mode tutorial <../examples/getting_started/pipeline_mode.ipynb>`_.
