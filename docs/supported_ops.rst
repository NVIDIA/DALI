Operations
==========

.. currentmodule:: nvidia.dali

In this section, we include a comprehensive list of all of DALI's supported operations.

Operator API vs. Functional API
-------------------------------

DALI's operations can be used in a DALI pipeline with two alternative APIs: The Operator API, and the Functional API (Recommended).

The Functional API is designed to simplify the usage of DALI operators in a psuedo-imperative way.
It exposes operators as functions, with the same name as the operator class, but converted
to snake_case - for example :class:`ops.FileReader` will be exposed as :func:`fn.file_reader`.

Operator API
^^^^^^^^^^^^

Operator API operations or "operators" are contained in the ``dali.ops`` module and their names are camel cased, for example ``dali.ops.ImageDecoder``.

The operator definition (``__init__``) and use in a DALI graph (``__call__``) are separate steps.
When using this approach, keyword arguments can be provided during instantiation (``__init__``) or in the graph definition (``__call__``) but positional
arguments can only be provided during graph definition.

Here is an example of a typical pipeline snippet using the operator API::

    import nvidia.dali as dali

    reader = dali.ops.FileReader(file_root='./my_file_root')
    decoder = dali.ops.ImageDecoder(device='mixed')
    rotate = dali.ops.Rotate()
    resize = dali.ops.Resize(resize_x=300, resize_y=300)
    rng = dali.ops.random.Uniform(range=(-45, 45))

    pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
    with pipe:
        files, labels = reader()
        images = decoder(files)
        angle = rng()
        images = rotate(images, angle=angle)
        images = resize(images)
        pipe.set_outputs(images, labels)

    pipe.build()
    outputs = pipe.run()


Functional API
^^^^^^^^^^^^^^
Functional API operations or "functions" are contained in the ``dali.fn`` module and their names are snake cased, for example ``dali.fn.image_decoder``.

Both keyword and positional arguments are provided during the graph definition, in a pseudo-imperative fashion. The use of functional API does not change
other aspects of pipeline definition - the functions still operate on and return :class:`pipeline.DataNode` objects.

Here is the same pipeline as above, written using the functional API::

    import nvidia.dali as dali

    pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
    with pipe:
        files, labels = dali.fn.file_reader(file_root = "./my_file_root")
        images = dali.fn.image_decoder(files, device = "mixed")
        images = dali.fn.rotate(images, angle = dali.fn.random.uniform(range=(-45,45)))
        images = dali.fn.resize(images, resize_x = 300, resize_y = 300)
        pipe.set_outputs(images, labels)

    pipe.build()
    outputs = pipe.run()

It is worth noting that both approaches are equivalent in terms of functionality and performance.

Interoperability between Operator and Functional API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Functional API is, for the major part, only a wrapper around operator objects - as such, it is
inherently compatible with the object-based API. The following example mixes the two,
using object API to pre-configure a file reader and a resize operator::


    pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
    reader = dali.ops.FileReader(file_root = ".")
    resize = dali.ops.Resize(device = "gpu", resize_x = 300, resize_y = 300)

    with pipe:
        files, labels = reader()
        images = dali.fn.image_decoder(files, device = "mixed")
        images = dali.fn.rotate(images, angle = dali.fn.random.uniform(range=(-45,45)))
        images = resize(images)
        pipe.set_outputs(images, labels)

    pipe.build()
    outputs = pipe.run()

Support table
-------------

The following table lists all available functions/operators, including information about 
the devices on which they can be executed:

.. include:: op_inclusion

Notes:

- | **CPU** operator means that the operator can be scheduled on the CPU.
  | The outputs of CPU operators may be used as regular inputs and to provide per-sample parameters
    for other operators through tensor arguments.
- **GPU** operator means that the operator can be scheduled on the GPU. Their outputs can only be
  used as
  regular inputs for other GPU operators and pipeline outputs.
- **Mixed** operator means that the operator accepts CPU inputs and produces GPU outputs.
- **Sequences** means that the operator can produce or accept as an input a sequence
  (for example, a video).
- **Volumetric** means that the operator supports 3D data processing.

Functions Documentation
-----------------------

.. include:: fn_autodoc

Operators Documentation
-----------------------

.. note::
    The names of the positional arguments for ``__call__`` operator (**parameters**) are provided
    only for documentation purposes and cannot be used as keyword arguments.

.. note::
    Keyword arguments can be provided to the class constructor and or to the ``__call__``
    operator.

.. include:: op_autodoc


nvidia.dali.plugin.pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:

Mathematical expressions
------------------------
DALI allows you to use regular Python arithmetic operations and other mathematical functions in
the :meth:`~nvidia.dali.pipeline.Pipeline.define_graph` method on the values that are returned
from invoking other operators. Full documentation can be found in the section dedicated to
:ref:`mathematical expressions`.
