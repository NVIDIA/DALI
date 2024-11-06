.. _datanode indexing:

Indexing and Slicing
====================

.. currentmodule:: nvidia.dali

DALI data nodes can be indexed and sliced using familiar Python syntax for array indexing ``x[sel]``
where ``x`` is a DALI data node and ``sel`` is the *selection*. The selection can be an *index* or
a *slice*.

Indexing
~~~~~~~~

The simplest case is scalar indexing with a constant index::

    images = fn.decoders.image(...)
    sizes = fn.sizes(images)  # height, width, channels

    height = sizes[0]
    width  = sizes[1]

The snippet above extracts the width and height from a 3-element tensor representing image size.

.. note::
    The batch dimension is implicit and cannot be indexed. In this example, the indices
    are broadcast to the whole batch. See :ref:`Indexing with run-time values` for per-sample
    indexing.

    See :func:`fn.permute_batch` for an operator which can access data from a different sample
    in the batch.

Indexing from the end
~~~~~~~~~~~~~~~~~~~~~

Negative indices can be used to index the tensor starting from the end. The index of -1 denotes
the last element::

    channels = sizes[-1]   # channels go last
    widths = sizes[-2]     # widths are the innermost dimension after channels

Indexing with run-time values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing with a constant is often insufficient. With DALI, you can use a result of other
computations to access tensor elements. In the example below, we use a run-time defined index to
access an element at a random position within a tensor::

    raw_files = fn.readers.file(...)
    length = raw_files.shape()[0]

    # calculate a random index from 0 to file_length-1
    random_01 = fn.random.uniform(range=(0, 1))  # random numbers in range [0..1)
    index = fn.floor(random_01 * length)  # calculate indices from [0..length)
    # cast the index to integer - required for indexing
    index = fn.cast(index, dtype=dali.types.INT64)

    # extract a random byte
    random_byte = raw_files[index]

Here, a byte at random index will be extracted from each sample. ``index`` is a data node which
represents a batch of scalar values, one per sample. Each of these values is used as the index
for the respective sample in ``raw_files``.

.. note::
    The index must be a result of a CPU operator.


Slicing
~~~~~~~

To extract multiple values (or slices), the Python list slicing syntax can be used::

    header = raw_files[:16]  # extract 16-byte headers from files in the batch

If the start of the slice is omitted, the slice starts at 0. If the end is omitted, the slice
ends at the end of given axis. Negative indices can be used for both start and end of the slice.
Either end can be a constant, a run-time value (a DataNode) or can be skipped.

Multidimensional selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

For multidimensional data, you can specify multiple, comma-separated selections.
If a selection is an index, the corresponding dimension is removed from the output::

    images = fn.decoders.image(jpegs, device="mixed")  # RGB images in HWC layout
    red =   images[:,:,0]
    green = images[:,:,1]
    blue =  images[:,:,2]

The ``red``, ``green``, ``blue`` are 2D tensors.

Slicing keeps the sliced dimensions even if the length of the slice is 1::

    green_with_channel = images[:,:,1:2]  # the last dimension is kept

When indexing and slicing multidimensional data, the trailing dimensions can be omitted. This is
equivalent to passing a full-range slice to all trailing dimensions::

    wide = letterboxed[20:-20,:,:]   # slice height, keep width and channels
    wide = letterboxed[20:-20,:]     # this line is equivalent to the previous one
    wide = letterboxed[20:-20]       # ...and so is this one

.. note::
    See also :func:`fn.crop` and :func:`fn.slice` for operations tailored for image processing.

Strided slices
~~~~~~~~~~~~~~

Striding in the positive and negative direction can also be achieved with the same
semantics as numpy arrays. This can be done over multiple dimensions.

    reversed = array[::-1]
    every_second = array[::2]
    every_second_reversed = array[::-2]
    quarter_resolution = image[::2, ::2]

Adding dimensions
~~~~~~~~~~~~~~~~~

A special value ``dali.newaxis`` can be used as an index. This value creates a new dimension of
size 1 in the output::

    trailing_channel = grayscale[:,:,dali.newaxis]
    leading_channel = grayscale[dali.newaxis]

Layout specifiers
~~~~~~~~~~~~~~~~~

DALI tensors can have a :ref:`layout specifier<data layouts>` which affects how the data is
interpreted by some operators - typically, an image would have ``"HWC"`` layout.

When applying a scalar index to an axis, that axis is removed from the output along with the
layout name for this axis::

    image = ...             # layout HWC
    first_row = image[0]    # layout WC
    last_col = image[:,-1]  # layout HC
    red = image[:,:,0]      # layout HW


A name can be added to the newly created dimension by passing it to ``dali.newaxis``::

    image = ... # layout is HWC
    single_frame_video = image[dali.newaxis("F")]  # layout is FHWC

