.. _data layouts

Tensor Layout String format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DALI uses short strings (Python `str` type) to describe data layout in tensors, by assigning a
character to each of the dimensions present in the tensor shape. For example, shape=(400, 300, 3),
layout="HWC" means that the data is an image with 3 interleaved channels, 400 pixels of height and
300 pixels of width.

For TensorLists, the index in the list is not treated as a dimension (the number of sample in the
batch) and is not included in the layout.

Interpreting Tensor Layout Strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DALI allows you to process data of different nature (e.g. image, video, audio, volumetric images)
as well as different formats (e.g. RGB image in planar configuration vs. interleaved channels).
Typically, DALI operators can deal with different data formats and will behave in different way
depending on the nature of the input.

While we do not restrict the valid characters to be used in a tensor layout, DALI operators
assume a certain naming convention. Here is a list of commonly used dimension names:

============== ==============
   Name           Meaning
============== ==============
   H              Height
   W              Width
   C              Channels
   F              Frames
   D              Depth
============== ==============

Here are some examples of typically used layouts:

============== ======================
   Layout         Description
============== ======================
   HWC            Image (interleaved)
   CHW            Image (planar)
   DHWC           Volumetric Image (interleaved)
   CDHW           Volumetric Image (planar)
   FHWC           Video
============== ======================

For instance, a crop operation (`Crop` operator) receiving an input with interleaved layout
(`"HWC"`) will infer that it should crop on the first and second dimensions `(H, W)`. On the
other hand, if the input has a planar layout (`"CHW"`) the crop will take place on the second and
third dimensions instead.

Some operators inherently modify the layout of the data (e.g. `Transpose`), while others
propagate the same data layout to the output (e.g. `Normalize`).

The layout restrictions (if any) for each operator are available through the operator's
documentation.

It is worth to note that the user is responsible to explicitly fill in the layout information
when using `ExternalSource` API.
