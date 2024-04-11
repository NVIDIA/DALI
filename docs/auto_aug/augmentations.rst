Augmentation operations
=======================

In terms of the automatic augmentations, the augmentation is image processing function that meets
following requirements:

1. Its first argument is the input batch for the processing
2. The second argument is the parameter controlling the operation (for example angle of rotation).
3. It can take additional keyword arguments.
4. It is implemented in terms of :ref:`DALI operators <operation reference>`.
5. It is decorated with :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`

Here is an example of defining a simplified rotate augmentation::

    from nvidia.dali.auto_aug.core import augmentation
    from nvidia.dali import fn

    @augmentation(mag_range=(0, 30), randomly_negate=True)
    def rotate_aug(data, angle, fill_value=128, rotate_keep_size=True):
       return fn.rotate(data, angle=angle, fill_value=fill_value, keep_size=True)

Based on the existing augmentation, a new one, with adjusted parameters, can be created::

    rotate_aug_60 = rotate_aug.augmentation(mag_range=(0, 60), randomly_negate=False)

To learn more how to build a policy using augmentations listed here, check the documentation for
specific automatic augmentation scheme: :ref:`AutoAugment`, :ref:`RandAugment`,
or :ref:`TrivialAugment`.

Decorator
---------

.. autodecorator:: nvidia.dali.auto_aug.core.augmentation

.. currentmodule:: nvidia.dali.auto_aug._augmentation

.. class:: nvidia.dali.auto_aug.core._augmentation.Augmentation

    The result of decorating a function with :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`
    is an instance of class :meth:`~nvidia.dali.auto_aug.core._augmentation.Augmentation`.
    The class should not be instantiated directly, it needs to be created with the decorator.

    Once obtained, those objects become callables that can be used to specify a policy for
    :ref:`AutoAugment`, :ref:`RandAugment` or :ref:`TrivialAugment`.

    .. method:: def augmentation(self, mag_range, randomly_negate, mag_to_param, param_device, name) -> Augmentation

        You can call this method to create new
        :meth:`~nvidia.dali.auto_aug.core._augmentation.Augmentation` instance based on an existing
        one, with the parameters adjusted. All parameters are optional - those that were specified
        replace the ones that were previously passed to
        :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.

        :param mag_range: optional, see :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.
        :param randomly_negate: optional, see :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.
        :param mag_to_param: optional, see :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.
        :param param_device: optional, see :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.
        :param name: optional, see :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>`.


Augmentations
-------------

Here is a list of callable :meth:`~nvidia.dali.auto_aug.core._augmentation.Augmentation` instances
defined by DALI. Note that the ``mag_to_param``, ``param_device`` and ``name`` parameters were
ommitted from the :py:func:`@augmentation <nvidia.dali.auto_aug.core.augmentation>` decorator listing
for simplicity.

To adjust the range of parameter, use the ``augmentation`` method
on the existing :meth:`~nvidia.dali.auto_aug.core._augmentation.Augmentation` instance listed below,
for example::

    # Create a steeper sheer operation based on existing one
    steep_shear_x = shear_x.augmentation(mag_range=(0, 0.5), name="steep_shear_x")


.. currentmodule:: nvidia.dali.auto_aug.augmentations

.. function:: shear_x(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.shear` with ``shear_x`` factor using
:meth:`nvidia.dali.fn.warp_affine`.


.. code-block:: python

    @augmentation(mag_range=(0, 0.3), randomly_negate=True, ...)
    def shear_x(data, shear, fill_value=128, interp_type=None)


.. function:: shear_y(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.shear` with ``shear_y`` factor using
:meth:`nvidia.dali.fn.warp_affine`.

.. code-block:: python

    @augmentation(mag_range=(0, 0.3), randomly_negate=True, ...)
    def shear_y(data, shear, fill_value=128, interp_type=None)

.. function:: translate_x(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.translation` with shape-relative offset in x-axis using
:meth:`nvidia.dali.fn.warp_affine`.

.. code-block:: python

    @augmentation(mag_range=(0., 1.), randomly_negate=True, ...)
    def translate_x(data, rel_offset, shape, fill_value=128, interp_type=None)

.. function:: translate_x_no_shape(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.translation` with absolute offset in x-axis using
:meth:`nvidia.dali.fn.warp_affine`.

.. code-block:: python

    @augmentation(mag_range=(0, 250), randomly_negate=True, ...)
    def translate_x_no_shape(data, offset, fill_value=128, interp_type=None)

.. function:: translate_y(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.translation` with shape-relative offset in y-axis using
:meth:`nvidia.dali.fn.warp_affine`.

.. code-block:: python

    @augmentation(mag_range=(0., 1.), randomly_negate=True, ...)
    def translate_y(data, rel_offset, shape, fill_value=128, interp_type=None)

.. function:: translate_y_no_shape(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies :meth:`nvidia.dali.fn.transforms.translation` with absolute offset in y-axis using
:meth:`nvidia.dali.fn.warp_affine`.

.. code-block:: python

    @augmentation(mag_range=(0, 250), randomly_negate=True, ...)
    def translate_y_no_shape(data, offset, fill_value=128, interp_type=None)

.. function:: rotate(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Rotates the image using :meth:`nvidia.dali.fn.rotate`.

.. code-block:: python

    @augmentation(mag_range=(0, 30), randomly_negate=True)
    def rotate(data, angle, fill_value=128, interp_type=None, rotate_keep_size=True)

.. function:: brightness(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Adjusts the brightness with :meth:`nvidia.dali.fn.brightness`. The magnitude is mapped to a [0, 2]
parameter range.

.. code-block:: python

    @augmentation(mag_range=(0, 0.9), randomly_negate=True, ...)
    def brightness(data, parameter)

.. function:: contrast(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)


Adjusts the contrasts using a channel-weighted mean as a contrast center. The magnitude is mapped
to a [0, 2] parameter range.

.. code-block:: python

    @augmentation(mag_range=(0, 0.9), randomly_negate=True, ...)
    def contrast(data, parameter)

.. function:: color(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Adjusts the color with :meth:`nvidia.dali.fn.saturation`. The magnitude is mapped to a [0, 2]
parameter range.

.. code-block:: python

    @augmentation(mag_range=(0, 0.9), randomly_negate=True, ...)
    def color(data, parameter)


.. function:: sharpness(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

The outputs correspond to `PIL's ImageEnhance.Sharpness <https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpnessl>`_.

.. code-block:: python

    @augmentation(mag_range=(0, 0.9), randomly_negate=True, ...)
    def sharpness(data, kernel)


.. function:: posterize(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Posterizes the image by masking out the lower input bits.

.. code-block:: python

    @augmentation(mag_range=(0, 4), ...)
    def posterize(data, mask)

.. function:: solarize(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Inverts the pixels that lie below a threshold.

.. code-block:: python

    @augmentation(mag_range=(256, 0), ...)
    def solarize(data, threshold)

.. function:: solarize_add(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies the shift to the pixels of value lower than 128.

.. code-block:: python

    @augmentation(mag_range=(0, 110), ...)
    def solarize_add(data, shift)

.. function:: invert(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Inverts the image.

.. code-block:: python

    @augmentation
    def invert(data, _)

.. function:: equalize(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies histogram equalization using :ref:`nvidia.dali.fn.experimental.equalize`.

.. code-block:: python

    @augmentation
    def equalize(data, _)
        """
        DALI's equalize follows OpenCV's histogram equalization.
        The PIL uses slightly different formula when transforming histogram's
        cumulative sum into lookup table.

.. function:: auto_contrast(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Applies automatic contrast adjustment.

.. code-block:: python

    @augmentation
    def auto_contrast(data, _)

.. function:: identity(data, *, magnitude_bin=None, num_magnitude_bins=None, **kwargs)

Identity operation - no processing is applied.

.. code-block:: python

    @augmentation
    def identity(data, _)
