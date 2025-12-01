.. _auto_aug:

Automatic Augmentations
=======================
Automatic augmentations are family of probabilistic augmentation policies.
When running automatic augmentation policy, each sample is processed with augmentations (operations
that transform the image) selected randomly according to some probability distribution defined by
the policy. DALI implements automatic augmentations with the usage of
:ref:`conditional execution <conditional_execution>`.

The ``nvidia.dali.auto_aug`` module contains ready to use policies for the popular automatic
augmentations - :ref:`AutoAugment`, :ref:`RandAugment`, and :ref:`TrivialAugment` - that can be
directly used within the processing pipeline definition. It provides a set of utilities to customize
the existing policies and allows to define new ones.

To use one of the policies define the pipeline using the
:py:func:`@pipeline_def <nvidia.dali.pipeline_def>` decorator and set ``enable_conditionals``
to ``True``. Next, call the automatic augmentation function inside the pipeline. This example
applies the :ref:`AutoAugment` policy tuned for ImageNet::
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.auto_aug import auto_augment


    @pipeline_def(enable_conditionals=True)
    def training_pipe(data_dir, image_size):

        jpegs, labels = fn.readers.file(file_root=data_dir, ...)
        shapes = fn.peek_image_shape(jpegs)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        # Applies the AutoAugment policy for ImageNet
        augmented_images = auto_augment.auto_augment_image_net(images, shape=shapes)

        resized_images = fn.resize(augmented_images, size=[image_size, image_size])

        return resized_images, labels

:ref:`RandAugment` and :ref:`TrivialAugment` policies can be applied in similar fashion.


.. note::
    To see a full example of using automatic augmentations for training, see the
    :ref:`EfficientNet for PyTorch with DALI and AutoAugment <efficientnet_autoaugment>` example.

.. note::
    You can also read more about Automatic Augmentation in the blogpost:
    `Why Automatic Augmentation Matters <https://developer.nvidia.com/blog/why-automatic-augmentation-matters/>`_.

    It covers the importance of Automatic Augmentations, explains the usage and possible customization,
    and shows how DALI can improve performance as compared to other implementations.

.. currentmodule:: nvidia.dali.auto_aug

.. toctree::
   :maxdepth: 1
   :hidden:

   AutoAugment <auto_augment>
   RandAugment <rand_augment>
   TrivialAugment <trivial_augment>
   Augmentations reference <augmentations>


Automatic Augmentation Library Structure
----------------------------------------
The automatic augmentation library is built around several concepts:

* **augmentation** - the image processing operation. DALI provides a
  :ref:`list of common augmentations <Augmentation operations>` that are used in AutoAugment,
  RandAugment, and TrivialAugment, as well as API for customization of those operations.
  :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>` decorator
  can be used to implement new augmentations.
* **policy** - a collection of augmentations and parameters that describe how to apply them to the
  input images - both the probability of application as well as the "strength" of the operation.
  DALI provides predefined policies as well as a way to define new ones - check the documentation
  for specific automatic augmentation scheme.
* **apply operation** - a function that invokes a specified policy on a batch of images within the
  DALI pipeline.

To learn more about building or applying policies check the documentation for specific automatic
augmentation scheme: :ref:`AutoAugment`, :ref:`RandAugment`, or :ref:`TrivialAugment`.
