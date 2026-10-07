Functional API
==============

.. warning::
   The DALI TorchVision API is an experimental feature and is subject to change.

This page documents the functions currently implemented in the DALI TorchVision API. They are
drop-in replacements for their counterparts in :mod:`torchvision.transforms.v2.functional`.
For a general introduction, see :doc:`overview`. To add an operator that is not listed here,
see :doc:`custom_operators`. For the operator classes, see :doc:`object_api`.

Every function accepts the additional ``device`` argument (``"cpu"`` or ``"gpu"``).

.. currentmodule:: nvidia.dali.experimental.torchvision.v2.functional

Available in the :mod:`nvidia.dali.experimental.torchvision.v2.functional` module.

Geometry functions
^^^^^^^^^^^^^^^^^^
.. autofunction:: resize
.. autofunction:: center_crop
.. autofunction:: crop
.. autofunction:: resized_crop
.. autofunction:: pad
.. autofunction:: horizontal_flip
.. autofunction:: vertical_flip

Color and filtering functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: rgb_to_grayscale
.. autofunction:: to_grayscale
.. autofunction:: gaussian_blur
.. autofunction:: normalize

PIL and tensor conversion functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pil_to_tensor
.. autofunction:: to_tensor
.. autofunction:: to_pil_image

Image metadata
^^^^^^^^^^^^^^
.. autofunction:: get_dimensions
.. autofunction:: get_image_size
.. autofunction:: get_size
