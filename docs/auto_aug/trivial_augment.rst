TrivialAugment
~~~~~~~~~~~~~~

.. currentmodule:: nvidia.dali.auto_aug.rand_augment

TrivialAugment, as described in https://arxiv.org/abs/2103.10158, is an automatic augmentation scheme
that is parameter-free - it can be used without the search for optimal meta-parameters.

Each sample is processed with just one randomly selected
:ref:`augmentation <Augmentation operations>`. The magnitude bin for every augmentation is randomly
selected.

To use the TrivialAugment, import and call the
:meth:`~nvidia.dali.auto_aug.trivial_augment.trivial_augment_wide` inside the pipeline definition,
for example::

    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.auto_aug import trivial_augment

    @pipeline_def(enable_conditionals=True)
    def training_pipe(data_dir, image_size):

        jpegs, labels = fn.readers.file(file_root=data_dir, ...)
        shapes = fn.peek_image_shape(jpegs)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        augmented_images = trivial_augment.trivial_augment_wide(images, shape=shapes)

        resized_images = fn.resize(augmented_images, size=[image_size, image_size])

        return resized_images, labels

:meth:`~nvidia.dali.auto_aug.trivial_augment.trivial_augment_wide` uses a standard set of
:ref:`augmentation <Augmentation operations>`, as described in the paper. To use a custom version
of TrivialAugment see the :ref:`TrivialAugment API` section.

.. warning::

    You need to define the pipeline with the :py:func:`@pipeline_def <nvidia.dali.pipeline_def>`
    decorator and set ``enable_conditionals`` to ``True`` to use automatic augmentations.

TrivialAugment API
------------------

The standard set of :ref:`augmentations <Augmentation operations>` (TrivialAugment Wide) can be used
by invoking the :meth:`~nvidia.dali.auto_aug.trivial_augment.trivial_augment_wide` inside the
pipeline definition.

A TrivialAugment policy is a list of :ref:`augmentations <Augmentation operations>`.
To obtain the list for the TrivialAugment Wide use
:meth:`~nvidia.dali.auto_aug.trivial_augment.get_trivial_augment_wide_suite`.

To use a custom list of :ref:`augmentations <Augmentation operations>`, pass it as a first argument
to the :meth:`~nvidia.dali.auto_aug.trivial_augment.apply_trivial_augment` invoked inside
the pipeline definition.

.. autofunction:: nvidia.dali.auto_aug.trivial_augment.trivial_augment_wide

.. autofunction:: nvidia.dali.auto_aug.trivial_augment.apply_trivial_augment

.. autofunction:: nvidia.dali.auto_aug.trivial_augment.get_trivial_augment_wide_suite
