[Experimental] DALI Dynamic
===========================

**DALI Dynamic** extends NVIDIA DALI by introducing an imperative execution model with lazy evaluation.
It complements the existing graph-based pipeline execution, and its main goal is to enable seamless
integration into Python workflows. This results in easier debugging and opens a path to rapid 
prototyping of pre-processing pipelines.

.. code-block:: python

   import nvidia.dali.experimental.dynamic as ndd

   model = MyModel(...)
   flip_horizontal = True
   flip_vertical = False
   dataset = ndd.readers.File(file_root=images_dir)
   for batch in dataset.next_epoch(batch_size=16):
       img = ndd.decoders.image(batch, device="mixed")
       flipped = ndd.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
       model((flipped, img))

See :doc:`overview` to learn more.

.. toctree::
   :maxdepth: 2

   overview
   api_reference
   readers_reference
   ops_reference