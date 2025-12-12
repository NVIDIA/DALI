// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {
namespace imgcodec {

DALI_SCHEMA(ImgcodecDecoderAttr)
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
will occur during the pipeline execution.)code",
      16*1024*1024)
  .AddOptionalArg("device_memory_padding_jpeg2k",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG2k's device memory allocations, in bytes. This parameter helps to avoid
reallocation in nvJPEG2k when a larger image is encountered, and the internal buffer needs to be
reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates the necessary number of buffers
according to the hint provided. If the value is correctly selected, no additional allocations
will occur during the pipeline execution.)code",
      0)
  .AddOptionalArg("host_memory_padding",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG's host memory allocations, in bytes. This parameter helps to prevent
the reallocation in nvJPEG when a larger image is encountered, and the internal buffer needs
to be reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates two (because of double-buffering)
host-pinned buffers of the requested size per thread. If selected correctly, no additional
allocations will occur during the pipeline execution.)code",
      8*1024*1024)  // based on ImageNet heuristics (8MB)
  .AddOptionalArg("host_memory_padding_jpeg2k",
      R"code(Applies **only** to the ``mixed`` backend type.

The padding for nvJPEG2k's host memory allocations, in bytes. This parameter helps to prevent
the reallocation in nvJPEG2k when a larger image is encountered, and the internal buffer needs
to be reallocated to decode the image.

If a value greater than 0 is provided, the operator preallocates the necessary number of buffers
according to the hint provided. If the value is correctly selected, no additional
allocations will occur during the pipeline execution.)code",
      0)
  .AddOptionalArg("hw_decoder_load",
      R"code(The percentage of the image data to be processed by the HW JPEG decoder.

Applies **only** to the ``mixed`` backend type in NVIDIA Ampere GPU architecture.

Determines the percentage of the workload that will be offloaded to the hardware decoder,
if available. The optimal workload depends on the number of threads that are provided to
the DALI pipeline and should be found empirically. More details can be found at
https://developer.nvidia.com/blog/loading-data-fast-with-dali-and-new-jpeg-decoder-in-a100)code",
      0.90f)
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
  .AddOptionalArg("use_fast_idct",
      R"code(Enables fast IDCT in the libjpeg-turbo based CPU decoder, used when `device` is set
to "cpu" or when the it is set to "mixed" but the particular image can not be handled by
the GPU implementation.

According to the libjpeg-turbo documentation, decompression performance is improved by up to 14%
with little reduction in quality.)code",
      false)
  .AddOptionalArg("jpeg_fancy_upsampling",
      R"code(Make the ``mixed`` backend use the same chroma upsampling approach as the ``cpu`` one.

The option corresponds to the `JPEG fancy upsampling` available in libjpegturbo or
ImageMagick.)code",
      false)
  .AddOptionalTypeArg("dtype",
      R"code(Output data type of the image.

Values will be converted to the dynamic range of the requested type.)code",
      DALI_UINT8)
  .AddOptionalArg("adjust_orientation",
      R"code(Use EXIF orientation metadata to rectify the images)code",
      true)
  // deprecated and removed (ignored)
  .AddOptionalArg("split_stages", "", false)
  .DeprecateArg("split_stages", "1.0", false)
  .AddOptionalArg("use_chunk_allocator", "", false)
  .DeprecateArg("use_chunk_allocator", "1.0", false)
  .AddOptionalArg("memory_stats", "", false)
  .DeprecateArg("memory_stats", "1.36", false);

DALI_SCHEMA(experimental__decoders__Image)
  .DocStr(R"code(Decodes images.

Supported formats: JPEG, JPEG 2000, TIFF, PNG, BMP, PNM, PPM, PGM, PBM, WebP.

The output of the decoder is in *HWC* layout.

The implementation uses NVIDIA nvImageCodec to decode images.

.. note::
  GPU accelerated decoding is only available for a subset of the image formats (JPEG, and JPEG2000).
  For other formats, a CPU based decoder is used. For JPEG, a dedicated HW decoder will be used when
  available.

.. note::
  WebP decoding currently only supports the simple file format (lossy and lossless compression).
  For details on the different WebP file formats, see
  https://developers.google.com/speed/webp/docs/riff_container)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("CachedDecoderAttr");

DALI_SCHEMA(experimental__decoders__ImageCrop)
  .DocStr(R"code(Decodes images and extracts regions-of-interest (ROI) that are specified
by fixed window dimensions and variable anchors.

Supported formats: JPEG, JPEG 2000, TIFF, PNG, BMP, PNM, PPM, PGM, PBM, WebP.

The output of the decoder is in *HWC* layout.

The implementation uses NVIDIA nvImageCodec to decode images.

When possible, the operator uses the ROI decoding, reducing the decoding time and memory consumption.

.. note::
  GPU accelerated decoding is only available for a subset of the image formats (JPEG, and JPEG2000).
  For other formats, a CPU based decoder is used. For JPEG, a dedicated HW decoder will be used when
  available.

.. note::
  WebP decoding currently only supports the simple file format (lossy and lossless compression).
  For details on the different WebP file formats, see
  https://developers.google.com/speed/webp/docs/riff_container

)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("CropAttr");


DALI_SCHEMA(experimental__decoders__ImageSlice)
  .DocStr(R"code(Decodes images and extracts regions of interest.

Supported formats: JPEG, JPEG 2000, TIFF, PNG, BMP, PNM, PPM, PGM, PBM, WebP.

The output of the decoder is in *HWC* layout.

The implementation uses NVIDIA nvImageCodec to decode images.

The slice can be specified by proving the start and end coordinates, or start coordinates
and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms.

The slice arguments can be specified by the following named arguments:

#. `start`: Slice start coordinates (absolute)
#. `rel_start`: Slice start coordinates (relative)
#. `end`: Slice end coordinates (absolute)
#. `rel_end`: Slice end coordinates (relative)
#. `shape`: Slice shape (absolute)
#. `rel_shape`: Slice shape (relative)

The slice can be configured by providing start and end coordinates or start and shape.
Relative and absolute arguments can be mixed (for example, `rel_start` can be used with `shape`)
as long as start and shape or end are uniquely defined.

Alternatively, two extra positional inputs can be provided, specifying `__anchor` and `__shape`.
When using positional inputs, two extra boolean arguments `normalized_anchor`/`normalized_shape`
can be used to specify the nature of the arguments provided. Using positional inputs for anchor
and shape is incompatible with the named arguments specified above.

The slice arguments should provide as many dimensions as specified by the `axis_names` or `axes`
arguments.

By default, the :meth:`nvidia.dali.fn.decoders.image_slice` operator uses normalized coordinates
and "WH" order for the slice arguments.

When possible, the operator uses the ROI decoding, reducing the decoding time and memory consumption.

.. note::
  GPU accelerated decoding is only available for a subset of the image formats (JPEG, and JPEG2000).
  For other formats, a CPU based decoder is used. For JPEG, a dedicated HW decoder will be used when
  available.

.. note::
  WebP decoding currently only supports the simple file format (lossy and lossless compression).
  For details on the different WebP file formats, see
  https://developers.google.com/speed/webp/docs/riff_container

)code")
  .NumInput(1, 3)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("SliceAttr")
  .InputDox(0, "data", "TensorList", R"code(Batch that contains the input data.)code")
  .InputDox(1, "anchor", "1D TensorList of float or int",
            R"code(Input that contains normalized or absolute coordinates for the starting
point of the slice (x0, x1, x2, …).

Integer coordinates are interpreted as absolute coordinates, while float coordinates can be
interpreted as absolute or relative coordinates, depending on the value of
`normalized_anchor`.)code")
  .InputDox(2, "shape", "1D TensorList of float or int",
            R"code(Input that contains normalized or absolute coordinates for the dimensions
of the slice (s0, s1, s2, …).

Integer coordinates are interpreted as absolute coordinates, while float coordinates can be
interpreted as absolute or relative coordinates, depending on the value of
`normalized_shape`.)code");


DALI_SCHEMA(experimental__decoders__ImageRandomCrop)
  .DocStr(R"code(Decodes images and randomly crops them.

Supported formats: JPEG, JPEG 2000, TIFF, PNG, BMP, PNM, PPM, PGM, PBM, WebP.

The output of the decoder is in *HWC* layout.

The implementation uses NVIDIA nvImageCodec to decode images.

The cropping window's area (relative to the entire image) and aspect ratio can be restricted to
a range of values specified by ``area`` and `aspect_ratio` arguments. respectively.

When possible, the operator uses the ROI decoding, reducing the decoding time and memory consumption.

.. note::
  GPU accelerated decoding is only available for a subset of the image formats (JPEG, and JPEG2000).
  For other formats, a CPU based decoder is used. For JPEG, a dedicated HW decoder will be used when
  available.

.. note::
  WebP decoding currently only supports the simple file format (lossy and lossless compression).
  For details on the different WebP file formats, see
  https://developers.google.com/speed/webp/docs/riff_container
)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("RandomCropAttr");

}  // namespace imgcodec
}  // namespace dali
