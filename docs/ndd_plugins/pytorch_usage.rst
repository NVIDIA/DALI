PyTorch Usage Guide
===================

.. currentmodule:: nvidia.dali.experimental.dynamic

DALI Dynamic provides interoperability with PyTorch, allowing you to use DALI operators
for high-performance data loading and preprocessing in your PyTorch workflows.

Basic usage
-----------

The following example reads images from disk, decodes and preprocesses them using DALI Dynamic
operators, and converts the results to PyTorch tensors:

.. code-block:: python

    import nvidia.dali.experimental.dynamic as ndd
    import nvidia.dali.types as types

    reader = ndd.readers.File(file_root=image_dir)
    for jpegs, labels in reader.next_epoch(batch_size=16):
        images = ndd.decoders.image(jpegs, device="gpu")
        images = ndd.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        images_torch = images.torch()
        labels_torch = labels.gpu().torch()

The conversion from DALI tensors to PyTorch tensors uses
`DLPack <https://dmlc.github.io/dlpack/latest/>`_ under the hood to avoid copies when possible
See :meth:`Tensor.torch` and :meth:`Batch.torch` in the API reference for details.

Next steps
----------

- :doc:`pytorch_torchdata_nodes_api`: Build composable data loading pipelines with
  :mod:`torchdata.nodes`.
- :doc:`../examples/frameworks/pytorch_dynamic/pytorch_training`: End-to-end training example
  using DALI Dynamic with PyTorch.
