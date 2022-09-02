// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/decoder.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/data/views.h"
#include "dali/imgcodec/image_decoder.h"

#include "dali/operators/imgcodec/operator_utils.h"

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
      false)
  .DeprecateArg("memory_stats")  // deprecated in imgcodec
  .AddOptionalTypeArg("dtype",
      R"code(Output data type of the image.

Values will be converted to the dynamic range of the requested type.)code",
      DALI_UINT8)
  .AddOptionalArg("adjust_orientation",
      R"code(Use EXIF orientation metadata to rectify the images)code",
      true);

DALI_SCHEMA(experimental__decoders__Image)
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
  https://developers.google.com/speed/webp/docs/riff_container)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr");

DALI_SCHEMA(experimental__decoders__ImageCrop)
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
  consider using separate ``decoders.image`` and ``crop`` operators.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("CropAttr");


DALI_SCHEMA(experimental__decoders__ImageSlice)
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
  consider using separate ``decoders.image`` and ``slice`` operators.)code")
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
``normalized_anchor``.)code")
  .InputDox(2, "shape", "1D TensorList of float or int",
            R"code(Input that contains normalized or absolute coordinates for the dimensions
of the slice (s0, s1, s2, …).

Integer coordinates are interpreted as absolute coordinates, while float coordinates can be
interpreted as absolute or relative coordinates, depending on the value of
``normalized_shape``.)code");


DALI_SCHEMA(experimental__decoders__ImageRandomCrop)
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
  EXIF orientation metadata is used to rectify the image.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddParent("ImgcodecDecoderAttr")
  .AddParent("RandomCropAttr");


ImgcodecHostDecoder::ImgcodecHostDecoder(const OpSpec &spec)
    : DecoderBase(spec) {
  device_id_ = CPU_ONLY_DEVICE_ID;
}

bool ImgcodecHostDecoder::SetupImpl(std::vector<OutputDesc> &output_descs,
                            const workspace_t<CPUBackend> &ws) {
  SetupShapes(spec_, ws, output_descs, ws.GetThreadPool());
  return true;
}

void ImgcodecHostDecoder::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  output.SetLayout("HWC");

  auto *decoder = GetDecoderInstance();

  int nsamples = input.shape().num_samples();

  DecodeContext ctx;
  ctx.tp = &ws.GetThreadPool();

  BatchVector<ImageSource> srcs;
  BatchVector<ImageSource *> src_ptrs;
  srcs.resize(nsamples);
  src_ptrs.resize(nsamples);

  for (int i = 0; i < nsamples; i++) {
    srcs[i] = SampleAsImageSource(input[i], input.GetMeta(i).GetSourceInfo());
    src_ptrs[i] = &srcs[i];
  }

  auto results = decoder->Decode(ctx, output, make_span(src_ptrs), opts_, make_span(rois_));
  for (const auto &result : results) {
    if (!result.success) {
      std::rethrow_exception(result.exception);
    }
  }
}

ImgcodecMixedDecoder::ImgcodecMixedDecoder(const OpSpec &spec)
    : DecoderBase(spec),
    thread_pool_{spec.GetArgument<int>("num_threads"), device_id_,
                 spec.GetArgument<bool>("affine"), "MixedDecoder"} {
}


bool ImgcodecMixedDecoder::SetupImpl(std::vector<OutputDesc> &output_descs,
                                     const workspace_t<MixedBackend> &ws) {
  SetupShapes(spec_, ws, output_descs, thread_pool_);
  return true;
}

void ImgcodecMixedDecoder::Run(workspace_t<MixedBackend> &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout("HWC");

  auto *decoder = GetDecoderInstance();

  int nsamples = input.shape().num_samples();

  DecodeContext ctx;
  ctx.tp = &thread_pool_;
  ctx.stream = ws.stream();

  BatchVector<ImageSource> srcs;
  BatchVector<ImageSource *> src_ptrs;
  srcs.resize(nsamples);
  src_ptrs.resize(nsamples);

  for (int i = 0; i < nsamples; i++) {
    srcs[i] = SampleAsImageSource(input[i], input.GetMeta(i).GetSourceInfo());
    src_ptrs[i] = &srcs[i];
  }

  auto results = decoder->Decode(ctx, output, make_span(src_ptrs), opts_, make_span(rois_));
  for (const auto &result : results) {
    if (!result.success) {
      std::rethrow_exception(result.exception);
    }
  }
}


using ImgcodecHostDecoderCrop = WithCropAttr<ImgcodecHostDecoder, CPUBackend>;
using ImgcodecHostDecoderSlice = WithSliceAttr<ImgcodecHostDecoder, CPUBackend>;
using ImgcodecHostDecoderRandomCrop = WithRandomCropAttr<ImgcodecHostDecoder, CPUBackend>;

using ImgcodecMixedDecoderCrop = WithCropAttr<ImgcodecMixedDecoder, MixedBackend>;
using ImgcodecMixedDecoderSlice = WithSliceAttr<ImgcodecMixedDecoder, MixedBackend>;
using ImgcodecMixedDecoderRandomCrop = WithRandomCropAttr<ImgcodecMixedDecoder, MixedBackend>;

DALI_REGISTER_OPERATOR(experimental__decoders__Image, ImgcodecHostDecoder, CPU);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageCrop, ImgcodecHostDecoderCrop, CPU);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageSlice, ImgcodecHostDecoderSlice, CPU);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageRandomCrop, ImgcodecHostDecoderRandomCrop, CPU);

DALI_REGISTER_OPERATOR(experimental__decoders__Image, ImgcodecMixedDecoder, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageCrop, ImgcodecMixedDecoderCrop, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageSlice, ImgcodecMixedDecoderSlice, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageRandomCrop,
                       ImgcodecMixedDecoderRandomCrop, Mixed);

}  // namespace imgcodec
}  // namespace dali
