.. _functional api:

Functional API
==============

.. currentmodule:: nvidia.dali

Quick start
-----------

.. warning::
    This API is experimental and subject to change without notice!

Functional API is designed to simplify the usage of DALI operators in a psuedo-imperative way.
It exposes operators as functions, with ths same name as the operator class, but converted
to snake_case - for example :class:`ops.FileReader` will be exposed as :func:`fn.file_reader`.

Example::

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

The use of functional API does not change other aspects of pipeline definition - the functions
still operate on and return :class:`pipeline.DataNode` objects.

Interoperability with operator objects
--------------------------------------
Functional API is, for the major part, only a wrapper around operator objects - as such, it is
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

Functions
---------

.. include:: fn_autodoc
