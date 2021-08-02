// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/pipeline/operator/operator.h"
#include "dali/operators/decoder/host/host_decoder.h"

namespace dali {

// ImageDecoder common attributes (does not include Cached decoder attributes which are present in
// ImageDecoder but not on the fused variants)
DALI_SCHEMA(ImageDecoderAttr)
  .DocStr(R"code(Image decoder common attributes)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("output_type",
      R"code(The color space of the output image.

Note: When decoding to YCbCr, the image will be decoded to RGB and then converted to YCbCr,
following the YCbCr definition from ITU-R BT.601.
)code",
      DALI_RGB)
  .AddOptionalArg("hybrid_huffman_threshold",
      R"code(Applies **only** to the ``mixed`` backend type.

Images with a total number of pixels (``height * width``) that is higher than this threshold will
use the nvJPEG hybrid Huffman decoder. Images that have fewer pixels will use the nvJPEG host-side
Huffman decoder.

.. note::
  Hybrid Huffman decoder still largely uses the CPU.)code",
      1000u*1000u)
  .AddOptionalArg("device_memory_padding",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG's device memory allocations, in bytes. This parameter helps to avoid
reallocation in nvJPEG when a larger image is encountered, and the internal buffer needs to be
reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates one device buffer of the
requested size per thread. If the value is correctly selected, no additional allocations
will occur during the pipeline execution. One way to find the ideal value is to do a complete
run over the dataset with the ``memory_stats`` argument set to True and then copy the largest
allocation value that was printed in the statistics.)code",
      16*1024*1024)
  .AddOptionalArg("device_memory_padding_jpeg2k",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG2k's device memory allocations, in bytes. This parameter helps to avoid
reallocation in nvJPEG2k when a larger image is encountered, and the internal buffer needs to be
reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates the necessary number of buffers
according to the hint provided. If the value is correctly selected, no additional allocations
will occur during the pipeline execution. One way to find the ideal value is to do a complete
run over the dataset with the ``memory_stats`` argument set to True and then copy the largest
allocation value that was printed in the statistics.)code",
      0)
  .AddOptionalArg("host_memory_padding",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG's host memory allocations, in bytes. This parameter helps to prevent
the reallocation in nvJPEG when a larger image is encountered, and the internal buffer needs
to be reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates two (because of double-buffering)
host-pinned buffers of the requested size per thread. If selected correctly, no additional
allocations will occur during the pipeline execution. One way to find the ideal value is to
do a complete run over the dataset with the ``memory_stats`` argument set to True, and then copy
the largest allocation value that is printed in the statistics.)code",
      8*1024*1024)  // based on ImageNet heuristics (8MB)
  .AddOptionalArg("host_memory_padding_jpeg2k",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG2k's host memory allocations, in bytes. This parameter helps to prevent
the reallocation in nvJPEG2k when a larger image is encountered, and the internal buffer needs
to be reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates the necessary number of buffers
according to the hint provided. If the value is correctly selected, no additional
allocations will occur during the pipeline execution. One way to find the ideal value is to
do a complete run over the dataset with the ``memory_stats`` argument set to True, and then copy
the largest allocation value that is printed in the statistics.)code",
      0)
  .AddOptionalArg("hw_decoder_load",
      R"code(The percentage of the image data to be processed by the HW JPEG decoder.

Applies **only** to the ``mixed`` backend type in NVIDIA Ampere GPU architecture.

Determines the percentage of the workload that will be offloaded to the hardware decoder,
if available. The optimal workload depends on the number of threads that are provided to
the DALI pipeline and should be found empirically. More details can be found at
https://developer.nvidia.com/blog/loading-data-fast-with-dali-and-new-jpeg-decoder-in-a100)code",
      0.65f)
  .AddOptionalArg("preallocate_width_hint",
      R"code(Image width hint.

Applies **only** to the ``mixed`` backend type in NVIDIA Ampere GPU architecture.

The hint is used to preallocate memory for the HW JPEG decoder.)code",
      0)
  .AddOptionalArg("preallocate_height_hint",
      R"code(Image width hint.

Applies **only** to the ``mixed`` backend type in NVIDIA Ampere GPU architecture.

The hint is used to preallocate memory for the HW JPEG decoder.)code",
      0)
  .AddOptionalArg("affine",
      R"code(Applies **only** to the ``mixed`` backend type.

If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
Otherwise, the threads can be reassigned to any CPU core by the operating system.)code",
      true)
  .AddOptionalArg("split_stages",
      R"code(Applies **only** to the ``mixed`` backend type.

If True, the operator will be split into two sub-stages: a CPU and GPU one.)code",
      false)
  .DeprecateArg("split_stages")  // deprecated in DALI 1.0
  .AddOptionalArg("use_chunk_allocator",
      R"code(**Experimental**, applies **only** to the ``mixed`` backend type.

Uses the chunk pinned memory allocator and allocates a chunk of the
``batch_size * prefetch_queue_depth`` size during the construction and suballocates
them at runtime.)code",
      false)
  .DeprecateArg("use_chunk_allocator")  // deprecated in DALI 1.0
  .AddOptionalArg("use_fast_idct",
      R"code(Enables fast IDCT in the libjpeg-turbo based CPU decoder, used when ``device`` is set
to "cpu" or when the it is set to "mixed" but the particular image can not be handled by
the GPU implementation.

According to the libjpeg-turbo documentation, decompression performance is improved by up to 14%
with little reduction in quality.)code",
      false)
  .AddOptionalArg("memory_stats",
      R"code(Applies **only** to the ``mixed`` backend type.

Prints debug information about nvJPEG allocations. The information about the largest
allocation might be useful to determine suitable values for ``device_memory_padding`` and
``host_memory_padding`` for a dataset.

.. note::
  The statistics are global for the entire process, not per operator instance, and include
  the allocations made during construction if the padding hints are non-zero.
)code",
      false);

DALI_SCHEMA(decoders__Image)
  .DocStr(R"code(Decodes images.

For jpeg images, depending on the backend selected ("mixed" and "cpu"), the implementation uses
the *nvJPEG* library or *libjpeg-turbo*, respectively. Other image formats are decoded
with *OpenCV* or other specific libraries, such as *libtiff*.

If used with a ``mixed`` backend, and the hardware is available, the operator will use
a dedicated hardware decoder.

.. warning::
  Due to performance reasons, hardware decoder is disabled for driver < 455.x

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000, WebP.
Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.

.. note::
  WebP decoding currently only supports the simple file format (lossy and lossless compression).
  For details on the different WebP file formats, see
  https://developers.google.com/speed/webp/docs/riff_container

.. note::
  EXIF orientation metadata is disregarded.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CachedDecoderAttr");

// Fused

DALI_SCHEMA(decoders__ImageCrop)
  .DocStr(R"code(Decodes images and extracts regions-of-interest (ROI) that are specified
by fixed window dimensions and variable anchors.

When possible, the argument uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to reduce the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000, WebP.

.. note::
  JPEG 2000 region-of-interest (ROI) decoding is not accelerated on the GPU, and will use
  a CPU implementation regardless of the selected backend. For a GPU accelerated implementation,
  consider using separate ``decoders.image`` and ``crop`` operators.

.. note::
  EXIF orientation metadata is disregarded.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CropAttr");

DALI_SCHEMA(decoders__ImageRandomCrop)
  .DocStr(R"code(Decodes images and randomly crops them.

The cropping window's area (relative to the entire image) and aspect ratio can be restricted to
a range of values specified by ``area`` and ``aspect_ratio`` arguments, respectively.

When possible, the operator uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to reduce the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000, WebP.

.. note::
  JPEG 2000 region-of-interest (ROI) decoding is not accelerated on the GPU, and will use
  a CPU implementation regardless of the selected backend. For a GPU accelerated implementation,
  consider using separate ``decoders.image`` and ``random_crop`` operators.

.. note::
  EXIF orientation metadata is disregarded.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("RandomCropAttr");


DALI_SCHEMA(decoders__ImageSlice)
  .DocStr(R"code(Decodes images and extracts regions of interest.

The slice can be specified by proving the start and end coordinates, or start coordinates
and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms.

The slice arguments can be specified by the following named arguments:

#. ``start``: Slice start coordinates (absolute)
#. ``rel_start``: Slice start coordinates (relative)
#. ``end``: Slice end coordinates (absolute)
#. ``rel_end``: Slice end coordinates (relative)
#. ``shape``: Slice shape (absolute)
#. ``rel_shape``: Slice shape (relative)

The slice can be configured by providing start and end coordinates or start and shape.
Relative and absolute arguments can be mixed (for example, ``rel_start`` can be used with ``shape``)
as long as start and shape or end are uniquely defined.

Alternatively, two extra positional inputs can be provided, specifying ``anchor`` and ``shape``.
When using positional inputs, two extra boolean arguments ``normalized_anchor``/``normalized_shape``
can be used to specify the nature of the arguments provided. Using positional inputs for anchor
and shape is incompatible with the named arguments specified above.

The slice arguments should provide as many dimensions as specified by the ``axis_names`` or ``axes``
arguments.

By default, the :meth:`nvidia.dali.fn.decoders.image_slice` operator uses normalized coordinates
and "WH" order for the slice arguments.

When possible, the argument uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to optimize the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

The output of the decoder is in the *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000, WebP.

.. note::
  JPEG 2000 region-of-interest (ROI) decoding is not accelerated on the GPU, and will use
  a CPU implementation regardless of the selected backend. For a GPU accelerated implementation,
  consider using separate ``decoders.image`` and ``slice`` operators.

.. note::
  EXIF orientation metadata is disregarded.)code")
  .NumInput(1, 3)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("SliceAttr")
  .InputDox(0, "data", "TensorList", R"code(Batch that contains the input data.)code")
  .InputDox(1, "anchor", "1D TensorList of float or int",
            R"code(Input that contains normalized or absolute coordinates for the starting
point of the slice (x0, x1, x2, …).

Integer coordinates are interpreted as absolute coordinates, while float coordinates can be
interpreted as absolute or relative coordinates, depending on the value of
``normalized_anchor``.)code")
  .InputDox(2, "shape", "1D TensorList of float or int",
            R"code(Input that contains normalized or absolute coordinates for the dimensions
of the slice (s0, s1, s2, …).

Integer coordinates are interpreted as absolute coordinates, while float coordinates can be
interpreted as absolute or relative coordinates, depending on the value of
``normalized_shape``.)code");


// Deprecated aliases

DALI_SCHEMA(ImageDecoder)
    .DocStr("Legacy alias for :meth:`decoders.image`.")
    .NumInput(1)
    .NumOutput(1)
    .AddParent("decoders__Image")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "decoders__Image",
        R"code(In DALI 1.0 all decoders were moved into a dedicated :mod:`~nvidia.dali.fn.decoders`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0

// Fused

DALI_SCHEMA(ImageDecoderCrop)
    .DocStr("Legacy alias for :meth:`decoders.image_crop`.")
    .NumInput(1)
    .NumOutput(1)
    .AddParent("decoders__ImageCrop")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "decoders__ImageCrop",
        R"code(In DALI 1.0 all decoders were moved into a dedicated :mod:`~nvidia.dali.fn.decoders`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0

DALI_SCHEMA(ImageDecoderRandomCrop)
    .DocStr("Legacy alias for :meth:`decoders.image_random_crop`.")
    .NumInput(1)
    .NumOutput(1)
    .AddParent("decoders__ImageRandomCrop")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "decoders__ImageRandomCrop",
        R"code(In DALI 1.0 all decoders were moved into a dedicated :mod:`~nvidia.dali.fn.decoders`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0


DALI_SCHEMA(ImageDecoderSlice)
    .DocStr("Legacy alias for :meth:`decoders.image_slice`.")
    .NumInput(1, 3)
    .NumOutput(1)
    .AddParent("decoders__ImageSlice")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "decoders__ImageSlice",
        R"code(In DALI 1.0 all decoders were moved into a dedicated :mod:`~nvidia.dali.fn.decoders`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0

}  // namespace dali
