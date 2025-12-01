AutoAugment
~~~~~~~~~~~

.. currentmodule:: nvidia.dali.auto_aug.auto_augment

AutoAugment, as described in https://arxiv.org/abs/1805.09501, builds policies out of pairs
of :ref:`augmentations <Augmentation operations>` called subpolicies.
Each subpolicy specifies sequence of operations with the probability of application and the
magnitude parameter. When AutoAugment is used, for each sample a random subpolicy is selected
and applied.

To use the predefined policy that was discovered on ImageNet, import and invoke
:meth:`~nvidia.dali.auto_aug.auto_augment.auto_augment` inside the pipeline definition,
for example::

    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.auto_aug import auto_augment

    @pipeline_def(enable_conditionals=True)
    def training_pipe(data_dir, image_size):

        jpegs, labels = fn.readers.file(file_root=data_dir, ...)
        shapes = fn.peek_image_shape(jpegs)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        augmented_images = auto_augment.auto_augment(images, shape=shapes)

        resized_images = fn.resize(augmented_images, size=[image_size, image_size])

        return resized_images, labels

.. warning::

    You need to define the pipeline with the :py:func:`@pipeline_def <nvidia.dali.pipeline_def>`
    decorator and set ``enable_conditionals`` to ``True`` to use automatic augmentations.

Refer to :ref:`this <Building and invoking custom policies>` section to read more about
using custom policies.

Invoking predefined AutoAugment policies
----------------------------------------

To invoke one of the predefined policies use the following functions.

.. autofunction:: nvidia.dali.auto_aug.auto_augment.auto_augment

.. autofunction:: nvidia.dali.auto_aug.auto_augment.auto_augment_image_net

Building and invoking custom policies
-------------------------------------

DALI's AutoAugment implementation relies on :meth:`~nvidia.dali.auto_aug.core.Policy` class to
define the policies to execute, which can be invoked within the pipeline using
:meth:`~nvidia.dali.auto_aug.auto_augment.apply_auto_augment` function.

The best way is to wrap your policy creation into a function::

   from nvidia.dali.auto_aug import augmentations
   from nvidia.dali.auto_aug.core import Policy

   def my_custom_policy() -> Policy:
        """
        Creates a simple AutoAugment policy with 3 sub-policies using custom
        magnitude ranges.
        """

        shear_x = augmentations.shear_x.augmentation((0, 0.5), True)
        shear_y = augmentations.shear_y.augmentation((0, 0.5), True)
        rotate = augmentations.rotate.augmentation((0, 40), True)
        invert = augmentations.invert
        return Policy(
            name="SimplePolicy", num_magnitude_bins=11, sub_policies=[
                [(shear_x, 0.8, 7), (shear_y, 0.8, 4)],
                [(invert, 0.4, None), (rotate, 0.6, 8)],
                [(rotate, 0.6, 7), (shear_y, 0.6, 3)],
            ])

The tuple within the subpolicy definition specifies:

* the augmentation to use,
* the probability of applying that augmentation (if this subpolicy is selected),
* the magnitude to be used.

.. autoclass:: nvidia.dali.auto_aug.core.Policy
   :members:
   :special-members: __init__

.. autofunction:: nvidia.dali.auto_aug.auto_augment.apply_auto_augment

Accessing predefined policies
-----------------------------

To obtain the predefined policy definition refer to the following functions.

.. autofunction:: nvidia.dali.auto_aug.auto_augment.get_image_net_policy

.. autofunction:: nvidia.dali.auto_aug.auto_augment.get_reduced_cifar10_policy

.. autofunction:: nvidia.dali.auto_aug.auto_augment.get_svhn_policy

.. autofunction:: nvidia.dali.auto_aug.auto_augment.get_reduced_image_net_policy
