Operator Objects (Legacy)
=========================

In older versions of DALI, an object-oriented API was used to define operations instead of
a functional API. The use of the object API is discouraged now and its documentation is shown
here for reference purposes.

The legacy object "operators" are contained in the ``dali.ops`` module and their names are camel cased, instead of snake cased.
For example, ``dali.ops.ImageDecoder`` is the legacy counterpart of ``dali.fn.image_decoder``.

When using the operator object API, the definition of the operator is separated from its use in a 
DALI pipeline, which allows to set static arguments during instantiation.

Here is an example pipeline using the (recommended) functional API::

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

and the legacy implementation using the operator object API::

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

It is worth noting that the two APIs can be used together in a single pipeline. Here is an example of that::

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

Mapping to functional API
^^^^^^^^^^^^^^^^^^^^^^^^^

The following table shows the correspondence between the operations in the current functional API
and the legacy operator objects API.

.. include:: fn_to_op_table

Modules
^^^^^^^

.. include:: op_autodoc


nvidia.dali.plugin.pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:
