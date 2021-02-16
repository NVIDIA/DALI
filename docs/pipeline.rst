.. _pipeline:

Pipeline
========

In DALI, any data processing task has a central object called Pipeline. Pipeline object is an
instance of :class:`nvidia.dali.Pipeline` or a derived class. Pipeline encapsulates the
data processing graph and the execution engine.

You can define a DALI Pipeline in the following ways:

#. by implementing a function that uses DALI operators inside and decorating it with the
:meth:`pipeline_def` decorator
#. by instantiating :class:`Pipeline` object directly, building the graph and setting the pipeline
outputs with :meth:`Pipeline.set_outputs`
#. by inheriting from :class:`Pipeline` class and overriding :meth:`Pipeline.define_graph`
(this is the legacy way of defining DALI Pipelines)

.. currentmodule:: nvidia.dali

.. autoclass:: Pipeline
   :members:
   :special-members: __enter__, __exit__

Data Processing Graphs
----------------------

DALI pipeline is represented as a graph of operations. There are two kinds of nodes in the graph:

 * Operators - created on each call to an operator
 * Data nodes (see :class:`~nvidia.dali.pipeline.DataNode`) - represent outputs and inputs of operators; they are
   returned from calls to operators and passing them as inputs to other operators establishes
   connections in the graph.

Example::

    @pipeline_def  # create a pipeline with processing graph defined by the function below
    def my_pipeline():
        """ Create a pipeline which reads images and masks, decodes the images and returns them. """
        img_files, labels = fn.file_reader(file_root="image_dir", seed=1)
        mask_files, _ = fn.file_reader(file_root="mask_dir", seed=1)
        images = fn.image_decoder(img_files, device="mixed")
        masks  = fn.image_decoder(mask_files, device="mixed")
        return images, masks, labels

    pipe = my_pipeline(batch_size=4, num_threads=2, device_id=0)
    pipe.build()


The resulting graph is:

.. image:: images/two_readers.svg

Current Pipeline
----------------

Subgraphs that do not contribute to the pipeline output are automatically pruned.
If an operator has side effects (e.g. ``PythonFunction`` operator family), it cannot be invoked
without setting the current pipeline. Current pipeline is set implicitly when the graph is
defined inside derived pipelines' :meth:`Pipeline.define_graph` method.
Otherwise, it can be set using context manager (``with`` statement)::

    pipe = dali.Pipeline(batch_size=N, num_threads=3, device_id=0)
    with pipe:
        src = dali.ops.ExternalSource(my_source, num_outputs=2)
        a, b = src()
        pipe.set_outputs(a, b)

When creating a pipeline with :meth:`pipeline_def`, the function which defines the pipeline is
executed within the scope of the newly created pipeline. The following example is equivalent
to the previous one::

    @dali.pipeline_def(batch_size=N, num_threads=3, device_id=0)
    def my_pipe(my_source):
        return dali.fn.external_source(my_source, num_outputs=2)

    pipe = my_pipe(my_source)

.. autoclass:: Pipeline
   :members:
   :special-members: __enter__, __exit__


Pipeline Decorator
------------------
.. autodecorator:: pipeline_def


DataNode
--------
.. autoclass:: nvidia.dali.pipeline.DataNode
   :members:
