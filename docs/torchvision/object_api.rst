Object API
==========

.. warning::
   The DALI TorchVision API is an experimental feature and is subject to change.

This page documents the operator classes currently implemented in the DALI TorchVision API. They are
drop-in replacements for their counterparts in :mod:`torchvision.transforms.v2`.
For a general introduction, see :doc:`overview`. To add an operator that is not listed here,
see :doc:`custom_operators`. For the functional counterparts, see :doc:`functional_api`.

Every operator accepts the additional ``device`` argument (``"cpu"`` or ``"gpu"``).
:class:`nvidia.dali.experimental.torchvision.Compose` additionally accepts ``batch_size``.


.. currentmodule:: nvidia.dali.experimental.torchvision

Available in the :mod:`nvidia.dali.experimental.torchvision.v2` module.

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
