Operator Objects (Legacy)
=========================

In older versions of DALI, an object-oriented API was used to define operations instead of
a functional API. The use of the object API is discouraged now and its documentation is shown
here for reference purposes.

The legacy object "operators" are contained in the ``dali.ops`` module and their names are camel cased, instead of snake cased.
For example, ``dali.ops.CropMirrorNormalize`` is the legacy counterpart of ``dali.fn.crop_mirror_normalize``.

When using the operator object API, the definition of the operator is separated from its use in a
DALI pipeline, which allows to set static arguments during instantiation.

Here is an example pipeline using the (recommended) functional API::

    import nvidia.dali as dali

    pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
    with pipe:
        files, labels = dali.fn.readers.file(file_root = "./my_file_root")
        images = dali.fn.decoders.image(files, device = "mixed")
        images = dali.fn.rotate(images, angle = dali.fn.random.uniform(range=(-45,45)))
        images = dali.fn.resize(images, resize_x = 300, resize_y = 300)
        pipe.set_outputs(images, labels)

    outputs = pipe.run()

and the legacy implementation using the operator object API::

    import nvidia.dali as dali

    class CustomPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super(CustomPipe, self).__init__(batch_size, num_threads, device_id)
            self.reader = dali.ops.readers.File(file_root='./my_file_root')
            self.decoder = dali.ops.ImageDecoder(device='mixed')
            self.rotate = dali.ops.Rotate()
            self.resize = dali.ops.Resize(resize_x=300, resize_y=300)
            self.rng = dali.ops.random.Uniform(range=(-45, 45))

        def define_graph(self):
            files, labels = self.reader()
            images = self.decoder(files)
            images = self.rotate(images, angle=self.rng())
            images = self.resize(images)
            return images, labels

    pipe = CustomPipe(batch_size = 3, num_threads = 2, device_id = 0)
    outputs = pipe.run()

It is worth noting that the two APIs can be used together in a single pipeline. Here is an example of that::

    pipe = dali.pipeline.Pipeline(batch_size = 3, num_threads = 2, device_id = 0)
    reader = dali.ops.readers.File(file_root = ".")
    resize = dali.ops.Resize(device = "gpu", resize_x = 300, resize_y = 300)

    with pipe:
        files, labels = reader()
        images = dali.fn.decoders.image(files, device = "mixed")
        images = dali.fn.rotate(images, angle = dali.fn.random.uniform(range=(-45,45)))
        images = resize(images)
        pipe.set_outputs(images, labels)

    outputs = pipe.run()

Mapping to Functional API
^^^^^^^^^^^^^^^^^^^^^^^^^

The following table shows the correspondence between the operations in the current functional API
and the legacy operator objects API.

.. include:: operations/fn_to_op_table

Modules
^^^^^^^

.. include:: operations/op_autodoc


nvidia.dali.plugin.pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:

Compose
^^^^^^^
.. autoclass:: nvidia.dali.ops.Compose
   :members:
