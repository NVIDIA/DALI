RandAugment
~~~~~~~~~~~

.. currentmodule:: nvidia.dali.auto_aug.rand_augment

RandAugment, as described in https://arxiv.org/abs/1909.13719, is an automatic augmentation scheme
that simplified the :ref:`AutoAugment`.
For RandAugment the policy is just a list of :ref:`augmentations <Augmentation operations>`
with a search space limited to two parameters ``n`` and ``m``.

* ``n`` describes how many randomly selected augmentations should we apply to an input sample.
* ``m`` is a fixed magnitude used for all of the augmentations.

For example, to use **3** random operations for each sample, each with fixed magnitude **17**,
you can call :meth:`~nvidia.dali.auto_aug.rand_augment.rand_augment`, as follows::

    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.auto_aug import rand_augment

    @pipeline_def(enable_conditionals=True)
    def training_pipe(data_dir, image_size):

        jpegs, labels = fn.readers.file(file_root=data_dir, ...)
        shapes = fn.peek_image_shape(jpegs)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        augmented_images = rand_augment.rand_augment(images, shape=shapes, n=3, m=17)

        resized_images = fn.resize(augmented_images, size=[image_size, image_size])

        return resized_images, labels

The :meth:`~nvidia.dali.auto_aug.rand_augment.rand_augment` uses set of augmentations described in
the paper. To apply custom augmentations refer to
:ref:`this section <Invoking custom RandAugment policies>`.

.. warning::

    You need to define the pipeline with the :py:func:`@pipeline_def <nvidia.dali.pipeline_def>`
    decorator and set ``enable_conditionals`` to ``True`` to use automatic augmentations.

Invoking predefined RandAugment policies
----------------------------------------

To invoke the predefined RandAugment policy, use the following function.

.. autofunction:: nvidia.dali.auto_aug.rand_augment.rand_augment

Invoking custom RandAugment policies
------------------------------------

Thanks to the simpler nature of RandAugment, its policies are defined as lists of
:ref:`augmentations <Augmentation operations>`, that can be passed as a first argument to the
:meth:`~nvidia.dali.auto_aug.rand_augment.apply_rand_augment` when invoked inside a pipeline
definition.

.. autofunction:: nvidia.dali.auto_aug.rand_augment.apply_rand_augment

Accessing predefined policies
-----------------------------

To obtain the predefined policy definition refer to the following functions.

.. autofunction:: nvidia.dali.auto_aug.rand_augment.get_rand_augment_suite

.. autofunction:: nvidia.dali.auto_aug.rand_augment.get_rand_augment_non_monotonic_suite
