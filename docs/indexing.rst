.. _datanode indexing:

Indexing
========

.. currentmodule:: nvidia.dali

DALI data nodes can be indexed and sliced using familiar Python syntax for array indexing.

Scalar indexing
~~~~~~~~~~~~~~~

The simplest case is scalar indexing with a constant index::

    raw_files = fn.readers.file(...)
    third_bytes = raw_files[2]

The snippet above extracts the third (index 2) byte from the files.

.. note::
    The batch dimension is implicit and cannot be indexed. In this example, the index 2
    is broadcast to the whole batch. See the section :ref:`Run-time indices` for per-sample
    indexing.

Indexing from the end
~~~~~~~~~~~~~~~~~~~~~

Negative indices can be used to index the tensor starting from the end. The index of -1 denotes
the last element::

    last_bytes = raw_files[-1]
    penultimate_bytes = raw_files[-2]


Run-time indices
~~~~~~~~~~~~~~~~

Indexing by constant is often insufficient. With DALI, you can use a result of other computations
to access tensor elements::

    length = fn.shapes(raw_files)[0]
    index = fn.cast(fn.random.uniform(range=(0, 1)) * length, dtype=dali.types.INT64)
    random_byte = raw_files[index]

Here, a byte at random index will be extracted from each sample. ``index`` is a data node which
represents a batch of scalar values, one per sample. Each of these values is used as the index
for the respective sample in ``raw_files``.

.. note::
    The index must be a result of a CPU operator.


Ranges
~~~~~~

To extract multiple values (or slices), the Python list slicing systax can be used::

    header = raw_files[:16]

If the start of the slice is omitted, the slice starts at 0. If the end is omitted, the slice
ends at the end of given axis. Negative indices can be used for both start and end of the slice.
Either end can be a constant, a run-time value (a DataNode) or can be skipped.

Multidimensional indexing
~~~~~~~~~~~~~~~~~~~~~~~~~

The index can contain multiple axes, separated by comma. If an index is a scalar, the corresponding
dimension is removed from the output::

    image = fn.decoders.image(jpegs, device="mixed")  # RGB images in HWC layout
    red =   image[:,:,0]
    green = image[:,:,1]
    blue =  image[:,:,2]

The ``red``, ``green``, ``blue`` are 2D tensors.

Using ranges doesn't remove dimensions::

    letterboxed = image[20:-20,:,:]   # the result is still a 3D HWC tensor

.. note::
    See also :func:`fn.crop` and :func:`fn.slice` for operations tailored for image processing.

When indexing a Multidimensional tensor, the trailing dimensions can be skipped - in this case,
they are treated as if full range was specified::

    video = ... # a 4D tensor in FHWC layout
    fifth_frame = video[5]         # equivalent to...
    fifth_frame = video[5,:]       # ...equivalent to...
    fifth_frame = video[5,:,:]     # ...equivalent to...
    fifth_frame = video[5,:,:,:]

Strided slices
~~~~~~~~~~~~~~

Indexing with a custom step is not implemented.

Adding dimensions
~~~~~~~~~~~~~~~~~

A special value ``dali.newaxis`` can be used as an index. This value creates a new dimension of
size 1 in the output::

    trailing_channel = grayscale[:,:,dali.newaxis]
    leading_channel = grayscale[dali.newaxis]

Layout specifiers
~~~~~~~~~~~~~~~~~

DALI tensors can have a layout specifier which affects how the data is interpreted by some
operators - typically, an image would have ``"HWC"`` layout.

When applying a scalar index to an axis, that axis is removed from the output along with the
layout name for this axis::

    image = ...             # layout HWC
    first_row = image[0]    # layout WC
    last_col = image[:,-1]  # lauout HWC
    red = image[:,:,0]      # layout HW


A name can be added to the newly created dimension by passing it to ``dali.newaxis``::

    image = ... # layout is HWC
    single_frame_video = image[dali.newaxis("F")]  # layout is FHWC

