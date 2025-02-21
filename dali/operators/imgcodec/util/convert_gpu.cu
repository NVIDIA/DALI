// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_kernel.cuh"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/operators/imgcodec/util/convert.h"
#include "dali/operators/imgcodec/util/convert_gpu.h"

namespace dali {
namespace imgcodec {

namespace {

constexpr int kDims = 3;

template <class Output, class Input>
void LaunchSliceFlipNormalizePermutePad(Output *out, TensorLayout out_layout,
                                        TensorShape<kDims> out_shape, const Input *in,
                                        TensorLayout in_layout, TensorShape<kDims> in_shape,
                                        kernels::KernelContext ctx, const ROI &roi,
                                        nvimgcodecOrientation_t orientation, float multiplier) {
  std::vector<kernels::SliceFlipNormalizePermutePadArgs<kDims>> args_container;
  args_container.emplace_back(out_shape, in_shape);
  auto &args = args_container[0];

  bool swap_xy = orientation.rotated % 180 == 90;
  bool flip_x = orientation.rotated == 180 || orientation.rotated == 270;
  bool flip_y = orientation.rotated == 90 || orientation.rotated == 180;
  flip_x ^= orientation.flip_x;
  flip_y ^= orientation.flip_y;

  auto adjust_layout_letter = [swap_xy](char c) {
    // if swap_xy is true, this function swaps width and height
    return (c == 'H' || c == 'W') && swap_xy ? c ^ 'H' ^ 'W' : c;
  };

  args.channel_dim = in_layout.find('C');
  for (int i = 0; i < kDims; i++) {
    char out_dim_letter = adjust_layout_letter(out_layout[i]);
    char in_dim_letter = adjust_layout_letter(in_layout[i]);
    args.permuted_dims[i] = in_layout.find(out_dim_letter);
    args.shape[args.permuted_dims[i]] = out_shape[i];  // the kernel applies permutation to shape
    args.flip[i] = (in_dim_letter == 'W' && flip_x) || (in_dim_letter == 'H' && flip_y);
  }

  if (roi) {
    for (int i = 0; i < roi.begin.sample_dim(); i++) {
      args.anchor[args.permuted_dims[i]] = roi.begin[i];
    }
  }

  args.mean.push_back(0.0f);
  args.inv_stddev.push_back(multiplier);

  kernels::SliceFlipNormalizePermutePadGpu<Output, Input, 3> kernel;
  TensorListView<StorageGPU, Output, kDims> tlv_out(out, {out_shape});
  TensorListView<StorageGPU, const Input, kDims> tlv_in(in, {in_shape});
  kernel.Setup(ctx, tlv_in, args_container);
  kernel.Run(ctx, tlv_out, tlv_in, args_container);
}

template <class Output, class Input>
void ConvertGPUImpl(SampleView<GPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
                    ConstSampleView<GPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
                    cudaStream_t stream, const ROI &roi, nvimgcodecOrientation_t orientation,
                    float multiplier) {
  auto scratchpad = kernels::DynamicScratchpad(AccessOrder(stream));
  kernels::KernelContext ctx;
  ctx.gpu.stream = stream;
  ctx.scratchpad = &scratchpad;

  DALI_ENFORCE(out.shape().sample_dim() == kDims && in.shape().sample_dim() == kDims,
               make_string("Conversion is only supported for ", kDims, "-dimensional tensors"));

  DALI_ENFORCE(out_layout.is_permutation_of("HWC") && in_layout.is_permutation_of("HWC"),
               "Layouts must be a permutation of HWC layout");

  // Starting with converting the layout, colorspace will be converted later
  auto intermediate_shape = out.shape();
  int channel_dim = out_layout.find('C');
  intermediate_shape[channel_dim] = NumberOfChannels(in_format, intermediate_shape[channel_dim]);

  // Normalize by changing the multiplier
  multiplier *=
      ConvertNorm<float>(static_cast<Input>(1)) / ConvertNorm<float>(static_cast<Output>(1));

  if (out_format == DALI_ANY_DATA)
    out_format = in_format;
  // The Slice kernel doesn't support converting color space
  bool needs_processing = out_format != in_format;
  Output *slice_out = out.mutable_data<Output>();
  if (needs_processing) {
    auto size = volume(intermediate_shape);
    slice_out = scratchpad.Allocate<mm::memory_kind::device, Output>(size);
  }

  LaunchSliceFlipNormalizePermutePad(slice_out, out_layout, intermediate_shape, in.data<Input>(),
                                     in_layout, in.shape(), ctx, roi, orientation, multiplier);

  if (needs_processing) {
    DALI_ENFORCE(out_layout.find('C') == kDims - 1,
                 "Only channel last layout is supported when running color space conversion");

    auto npixels = out.shape()[0] * out.shape()[1];
    kernels::color::RunColorSpaceConversionKernel(out.mutable_data<Output>(), slice_out, out_format,
                                                  in_format, npixels, stream);
  }
}

}  // namespace

void ConvertGPU(SampleView<GPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
                ConstSampleView<GPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
                cudaStream_t stream, const ROI &roi, nvimgcodecOrientation_t orientation,
                float multiplier) {
  TYPE_SWITCH(out.type(), type2id, Output, (IMGCODEC_TYPES), (
    TYPE_SWITCH(in.type(), type2id, Input, (IMGCODEC_TYPES), (
      ConvertGPUImpl<Output, Input>(out, out_layout, out_format, in, in_layout, in_format,
                                 stream, roi, orientation, multiplier);
    ), DALI_FAIL(make_string("Unsupported input type: ", in.type())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported output type: ", out.type())));  // NOLINT
}

}  // namespace imgcodec
}  // namespace dali
