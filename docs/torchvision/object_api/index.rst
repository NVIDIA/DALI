Object API
==========

.. warning::
   The DALI TorchVision API is an experimental feature and is subject to change.

The object-oriented API mirrors :mod:`torchvision.transforms.v2`: instantiate operator classes and
chain them with :class:`~nvidia.dali.experimental.torchvision.Compose`, which builds and owns the
underlying DALI pipeline.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Examples
      :link: object_api
      :link-type: doc

      Using ``Compose`` with ``PIL.Image`` and ``torch.Tensor`` input, compared with TorchVision.

   .. grid-item-card:: Reference
      :link: reference
      :link-type: doc

      All operator classes currently available in the DALI TorchVision API.

.. toctree::
   :hidden:

   Examples <object_api>
   Reference <reference>
