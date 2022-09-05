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

#include <vector>
#include "dali/imgcodec/util/convert_gpu.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_kernel.cuh"

namespace dali {

template<> float16 max_value<float16>() {
  assert(false && "This overload shouldn't be called");
}

namespace imgcodec {

namespace {

constexpr int dims = 3;

template<class Output, class Input>
void LaunchSliceFlipNormalizePermutePad(
    Output *out, TensorLayout out_layout, TensorShape<dims> out_shape,
    const Input *in, TensorLayout in_layout, TensorShape<dims> in_shape,
    kernels::KernelContext ctx, const ROI &roi, float multiplier) {
  // this normalization only works if Output range is [-1, 1]
  static_assert(std::is_floating_point<Output>::value);
  if (std::is_integral<Input>::value)
    multiplier /= max_value<Input>();

  std::vector<kernels::SliceFlipNormalizePermutePadArgs<dims>> args_container;
  args_container.emplace_back(out_shape, in_shape);
  auto &args = args_container[0];

  args.channel_dim = in_layout.find('C');
  for (int i = 0; i < dims; i++) {
    args.permuted_dims[i] = in_layout.find(out_layout[i]);
    args.shape[args.permuted_dims[i]] = out_shape[i];
  }

  if (roi) {
    for (int i = 0; i < roi.begin.sample_dim(); i++) {
      args.anchor[args.permuted_dims[i]] = roi.begin[i];
    }
  }

  args.mean.push_back(0.0f);
  args.inv_stddev.push_back(multiplier);

  kernels::SliceFlipNormalizePermutePadGpu<Output, Input, 3> kernel;
  TensorListView<StorageGPU, Output, dims> tlv_out(out, {out_shape});
  TensorListView<StorageGPU, const Input, dims> tlv_in(in, {in_shape});
  kernel.Setup(ctx, tlv_in, args_container);
  kernel.Run(ctx, tlv_out, tlv_in, args_container);
}

template <typename Output, typename Input>
__global__ void convert_sat_norm_kernel(Output *out, const Input *in, int64_t size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  out[tid] = ConvertSatNorm<Output>(in[tid]);
}

template <typename Output, typename Input>
void LaunchConvertSatNorm(Output *out, const Input *in, size_t size, cudaStream_t stream) {
  size_t block_size = size < 1024 ? size : 1024;
  size_t num_blocks = (size + block_size - 1) / block_size;
  convert_sat_norm_kernel<<<num_blocks, block_size, 0, stream>>>(out, in, size);
}

template<class T>
T read_from_gpu(T *ptr) {
  T obj;
  CUDA_CALL(cudaMemcpy(&obj, ptr, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  return obj;
}

template<class Output, class Input>
void ConvertImpl(SampleView<GPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
                 ConstSampleView<GPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
                 cudaStream_t stream, const ROI &roi, float multiplier) {
  kernels::DynamicScratchpad scratchpad({}, AccessOrder(stream));
  kernels::KernelContext ctx;
  ctx.gpu.stream = stream;
  ctx.scratchpad = &scratchpad;

  DALI_ENFORCE(out.shape().sample_dim() == dims && in.shape().sample_dim() == dims,
               make_string("Conversion is only supported for ", dims, "-dimensional tensors"));

  DALI_ENFORCE(out_layout.is_permutation_of("HWC") && in_layout.is_permutation_of("HWC"),
               "Layouts must be a permutation of HWC layout");

  // Starting with converting the layout, colorspace will be converted later
  auto intermediate_shape = out.shape();
  int channel_dim = out_layout.find('C');
  intermediate_shape[channel_dim] = NumberOfChannels(in_format, intermediate_shape[channel_dim]);

  auto size = volume(intermediate_shape);
  auto buffer = scratchpad.Allocate<mm::memory_kind::device, float>(size);
  LaunchSliceFlipNormalizePermutePad(
    buffer, out_layout, intermediate_shape, in.data<Input>(), in_layout, in.shape(),
    ctx, roi, multiplier);

  if (out_format != in_format) {
    DALI_ENFORCE(out_layout.find('C') == dims - 1,
                 "Only channel last layout is supported when running color space conversion");

    auto npixels = out.shape()[0] * out.shape()[1];
    kernels::color::RunColorSpaceConversionKernel(
      out.mutable_data<Output>(), buffer, out_format, in_format, npixels, stream);
  } else {
    LaunchConvertSatNorm(out.mutable_data<Output>(), buffer, size, stream);
  }
}

}  // namespace

void Convert(SampleView<GPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
             ConstSampleView<GPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
             cudaStream_t stream, const ROI &roi, float multiplier) {
  TYPE_SWITCH(out.type(), type2id, Output, (IMGCODEC_TYPES), (
    TYPE_SWITCH(in.type(), type2id, Input, (IMGCODEC_TYPES), (
      ConvertImpl<Output, Input>(out, out_layout, out_format,
                                 in, in_layout, in_format,
                                 stream, roi, multiplier);
    ), DALI_FAIL(make_string("Unsupported input type: ", in.type())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported output type: ", out.type())));  // NOLINT
}

}  // namespace imgcodec
}  // namespace dali
