// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
      R"code(The color space of the output image.)code",
      DALI_RGB)
// TODO(janton): Remove this when we remove the old nvJPEGDecoder implementation (DALI-971)
#if !defined(NVJPEG_DECOUPLED_API)
  .AddOptionalArg("use_batched_decode",
      R"code(**`mixed` backend only** Use nvJPEG's batched decoding API.)code", false)
#endif
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
  .AddOptionalArg("affine",
      R"code(Applies **only** to the ``mixed`` backend type.

If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
Otherwise, the threads can be reassigned to any CPU core by the operating system.)code",
      true)
  .AddOptionalArg("split_stages",
      R"code(Applies **only** to the ``mixed`` backend type.

If True, the operator will be split into two sub-stages: a CPU and GPU one.)code",
      false)
  .AddOptionalArg("use_chunk_allocator",
      R"code(**Experimental**, applies **only** to the ``mixed`` backend type.

Uses the chunk pinned memory allocator and allocates a chunk of the
``batch_size * prefetch_queue_depth`` size during the construction and suballocates
them at runtime. When ``split_stages`` is false, this argument is ignored.)code",
      false)
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

DALI_SCHEMA(ImageDecoder)
  .DocStr(R"code(Decodes images.

For jpeg images, depending on the backend selected ("mixed" and "cpu"), the implementation uses
the *nvJPEG* library or *libjpeg-turbo*, respectively. Other image formats are decoded
with *OpenCV* or other specific libraries, such as *libtiff*.

If used with a ``mixed`` backend, and the hardware is available, the operator will use
a dedicated hardware decoder.

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000.
Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.)code")
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
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CachedDecoderAttr");

// Fused

DALI_SCHEMA(ImageDecoderCrop)
  .DocStr(R"code(Decodes images and extracts regions-of-interest (ROI) that are specified
by fixed window dimensions and variable anchors.

When possible, the argument uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to reduce the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

.. note::
  ROI decoding is currently not compatible with hardware-based decoding. Using
  :meth:`nvidia.dali.ops.ImageDecoderCrop` automatically disables hardware accelerated
  decoding. To use the hardware decoder, use the :meth:`nvidia.dali.ops.ImageDecoder` and
  :meth:`nvidia.dali.ops.Crop` operators instead.

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000.
Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("CropAttr");

DALI_SCHEMA(ImageDecoderRandomCrop)
  .DocStr(R"code(Decodes images and randomly crops them.

The cropping window's area (relative to the entire image) and aspect ratio can be restricted to
a range of values specified by ``area`` and ``aspect_ratio`` arguments, respectively.

When possible, the operator uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to reduce the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

.. note::
  ROI decoding is currently not compatible with hardware-based decoding. Using
  :meth:`nvidia.dali.ops.ImageDecoderRandomCrop` automatically disables hardware accelerated
  decoding. To use the hardware decoder, use the :meth:`nvidia.dali.ops.ImageDecoder` and
  :meth:`nvidia.dali.ops.RandomResizedCrop` operators instead.

The output of the decoder is in *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000.
Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("RandomCropAttr");


DALI_SCHEMA(ImageDecoderSlice)
  .DocStr(R"code(Decodes images and extracts regions of interest based on externally provided
anchors and shapes.

Inputs must be supplied as tensors in the following order:

* ``data`` that contains the input data.
* ``anchor`` that contains normalized or absolute coordinates, depending on the
  ``normalized_anchor`` value, for the starting point of the slice (x0, x1, x2, and so on),
* ``shape`` that contains normalized or absolute coordinates, depending on the
  ``normalized_shape`` value, for the dimensions of the slice (s0, s1, s2, and so on).

The anchor and shape coordinates must be within the interval [0.0, 1.0] for normalized
coordinates or within the image shape for the absolute coordinates. The ``anchor`` and ``shape``
inputs will provide as many dimensions as were specified with arguments ``axis_names`` or ``axes``.

By default, the :meth:`nvidia.dali.ops.ImageDecoderSlice` operator uses normalized coordinates
and "WH" order for the slice arguments.

When possible, the argument uses the ROI decoding APIs (for example, *libjpeg-turbo* and *nvJPEG*)
to optimize the decoding time and memory usage. When the ROI decoding is not supported for a given
image format, it will decode the entire image and crop the selected ROI.

.. note::
  ROI decoding is currently not compatible with hardware-based decoding. Using
  :meth:`nvidia.dali.ops.ImageDecoderSlice` automatically disables hardware accelerated decoding.
  To use the hardware decoder, use the :meth:`nvidia.dali.ops.ImageDecoder` and
  :meth:`nvidia.dali.ops.Slice` operators instead.

The output of the decoder is in the *HWC* layout.

Supported formats: JPG, BMP, PNG, TIFF, PNM, PPM, PGM, PBM, JPEG 2000.
Please note that GPU acceleration for JPEG 2000 decoding is only available for CUDA 11.)code")
  .NumInput(3)
  .NumOutput(1)
  .AddParent("ImageDecoderAttr")
  .AddParent("SliceAttr");

}  // namespace dali
