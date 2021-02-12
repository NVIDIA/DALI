Pipeline
========

In DALI, any data processing task has a central object called pipeline. Pipeline object is an
instance of :class:`nvidia.dali.pipeline.Pipeline` or a derived class. Pipeline encapsulates the
data processing graph and the execution engine.

There are two ways to define a DALI pipelines:

#. by inheriting from Pipeline class and overriding :meth:`Pipeline.define_graph`
#. by instantiating `Pipeline` directly, building the graph and setting the pipeline
   outputs with :meth:`Pipeline.set_outputs`

.. currentmodule:: nvidia.dali.pipeline

.. autoclass:: Pipeline
   :members:
   :special-members: __enter__, __exit__

Data Processing Graphs
----------------------

DALI pipeline is represented as a graph of operations. There are two kinds of nodes in the graph:

 * Operators - created on each call to an operator
 * Data nodes (see :class:`DataNode`) - represent outputs and inputs of operators; they are
   returned from calls to operators and passing them as inputs to other operators establishes
   connections in the graph.

Example::

    class MyPipeline(Pipeline):
        def define_graph(self):
            img_reader  = ops.FileReader(file_root = "image_dir", seed = 1)
            mask_reader = ops.FileReader(file_root = "mask_dir", seed = 1)
            img_files, labels = img_reader()  # creates an instance of `FileReader`
            mask_files, _ = mask_reader()     # creates another instance of `FileReader`
            decode = ops.ImageDecoder()
            images = decode(img_files)  # creates an instance of `ImageDecoder`
            masks  = decode(mask_files)   # creates another instance of `ImageDecoder`
            return [images, masks, labels]

    pipe = MyPipeline(batch_size = 4, num_threads = 2, device_id = 0)
    pipe.build()


The resulting graph is:

.. image:: images/two_readers.svg

Current Pipeline
----------------

Subgraphs that do not contribute to the pipeline output are automatically pruned.
If an operator has side effects (e.g. PythonFunction operator family), it cannot be invoked
without setting the current pipeline. Current pipeline is set implicitly when the graph is
defined inside derived pipeline's `define_graph` method. Otherwise, it can be set using context
manager (`with` statement)::

    pipe = dali.pipeline.Pipeline(batch_size = N, num_threads = 3, device_id = 0)
    with pipe:
        src = dali.ops.ExternalSource(my_source, num_outputs = 2)
        a, b = src()
        pipe.set_outputs(a, b)

DataNode
--------

.. autoclass:: DataNode
   :members:
