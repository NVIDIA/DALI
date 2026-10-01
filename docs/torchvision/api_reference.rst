Torchvision API Reference
=========================

.. warning::
   The DALI Torchvision API is an experimental feature and is subject to change.

This page documents the operators currently implemented in the DALI Torchvision API. They are
drop-in replacements for their counterparts in
`torchvision.transforms.v2 <https://docs.pytorch.org/vision/stable/transforms.html>`_.
For a general introduction, see :doc:`overview`. To add an operator that is not listed here,
see :doc:`custom_operators`.

Every operator accepts the additional ``device`` argument (``"cpu"`` or ``"gpu"``). ``Compose``
additionally accepts ``batch_size``.

Operator classes
----------------

.. currentmodule:: nvidia.dali.experimental.torchvision

Available in the :mod:`nvidia.dali.experimental.torchvision` module.

Composing transforms
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Compose
   :members:

Geometry transforms
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Resize
.. autoclass:: CenterCrop
.. autoclass:: RandomCrop
.. autoclass:: RandomResizedCrop
.. autoclass:: Pad
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomVerticalFlip

Color and filtering transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ColorJitter
.. autoclass:: Grayscale
.. autoclass:: RandomGrayscale
.. autoclass:: GaussianBlur
.. autoclass:: Normalize

Control flow
^^^^^^^^^^^^
.. autoclass:: RandomApply

Type conversion transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PILToTensor
.. autoclass:: ToPILImage
.. autoclass:: ToPureTensor

Enumerations
^^^^^^^^^^^^
.. autoclass:: InterpolationMode
   :members:
   :undoc-members:

Operator functions
------------------

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
