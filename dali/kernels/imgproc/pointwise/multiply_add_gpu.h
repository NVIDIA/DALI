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

#ifndef DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_GPU_H_
#define DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_GPU_H_

#include <vector>
#include "dali/core/span.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {
namespace kernels {
namespace multiply_add {


template <size_t ndims>
using Roi = Box<ndims, int>;


template <class OutputType, class InputType, int ndims>
struct SampleDescriptor {
  const InputType *in;
  OutputType *out;
  ivec<ndims - 1> in_pitch, out_pitch;
  float addend, multiplier;
};

/**
 * Flattens the TensorShape
 *
 * Flattened TensorShape in case of MultiplyAdd kernel is a TensorShape,
 * in which channel-dimension is removed. Instead, the one-before dimension is
 * multiplied by channel-dimension size.
 *
 * E.g. [640, 480, 3] -> [640, 1440]
 *
 * The reason is that MultiplyAdd calculations are channel-agnostic
 * (the same operation is applied for every channel), therefore BlockSetup
 * and SampleDescriptor don't need to know about channels.
 */
template <int ndims>
TensorShape<ndims - 1> FlattenChannels(const TensorShape<ndims> &shape) {
  static_assert(ndims >= 2, "If there are less than 2 dims, there's nothing to flatten...");
  TensorShape<ndims - 1> ret;
  for (int i = 0; i < shape.size() - 1; i++) {
    ret[i] = shape[i];
  }
  ret[shape.size() - 2] *= shape[shape.size() - 1];
  return ret;
}


/**
 * Convenient overload for TensorListShape case
 */
template <int ndims>
TensorListShape<ndims - 1> FlattenChannels(const TensorListShape<ndims> &shape) {
  static_assert(ndims >= 2, "If there are less than 2 dims, there's nothing to flatten...");
  TensorListShape<ndims - 1> ret(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    ret.set_tensor_shape(i, FlattenChannels<ndims>(shape[i]));
  }
  return ret;
}


template <int ndim>
ivec<ndim - 2> pitch_flatten_channels(const TensorShape<ndim> &shape) {
  ivec<ndim - 2> ret;
  int stride = shape[ndim - 1];  // channels
  for (int i = ndim - 2; i > 0; i--) {
    stride *= shape[i];
    ret[ndim - 2 - i] = stride;  // x dimension is at ret[0] - use reverse indexing
  }
  return ret;
}

/**
 * Note: Since the operation we perform is channel-agnostic (it is performed in the same
 * way for every channel), SampleDescriptor assumes, that sample is channel-agnostic. Therefore it
 * needs to be flattened
 * @see FlattenChannels
 */
template <class OutputType, class InputType, int ndims>
void CreateSampleDescriptors(
        span<SampleDescriptor<OutputType, InputType, ndims - 1>> out_descs,
        const OutListGPU<OutputType, ndims> &out,
        const InListGPU<InputType, ndims> &in,
        const std::vector<float> &addends, const std::vector<float> &multipliers) {
  assert(out_descs.size() >= in.num_samples());

  for (int i = 0; i < in.num_samples(); i++) {
    auto &sample = out_descs[i];
    sample.in = in[i].data;
    sample.out = out[i].data;

    sample.in_pitch = pitch_flatten_channels(in[i].shape);
    sample.out_pitch = pitch_flatten_channels(out[i].shape);

    sample.addend = addends[i];
    sample.multiplier = multipliers[i];
  }
}


template <class OutputType, class InputType, int ndims>
__global__ void
MultiplyAddKernel(const SampleDescriptor<OutputType, InputType, ndims> *samples,
                  const BlockDesc<ndims> *blocks) {
  static_assert(ndims == 2, "Function requires 2 dimensions in the input");
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  const auto *__restrict__ in = sample.in;
  auto *__restrict__ out = sample.out;

  for (int y = threadIdx.y + block.start.y; y < block.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
      out[y * sample.out_pitch.x + x] = ConvertSat<OutputType>(
              in[y * sample.in_pitch.x + x] * sample.multiplier + sample.addend);
    }
  }
}

}  // namespace multiply_add

template <typename OutputType, typename InputType, int ndims>
class MultiplyAddGpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;
  using BlockDesc = kernels::BlockDesc<spatial_dims>;

  using SampleDesc = multiply_add::SampleDescriptor<OutputType, InputType, spatial_dims>;
  std::vector<SampleDesc> sample_descriptors_;

 public:
  BlockSetup<spatial_dims, -1 /* No channel dimension, only spatial */> block_setup_;


  KernelRequirements Setup(
      KernelContext &context,
      const InListGPU<InputType, ndims> &in,
      const std::vector<float> &addends, const std::vector<float> &multipliers,
      const std::vector<Roi<spatial_dims>> &rois = {}) {
    DALI_ENFORCE(rois.empty() || rois.size() == static_cast<size_t>(in.num_samples()),
                 "Provide ROIs either for all or none input tensors");
    DALI_ENFORCE([=]() -> bool {
        for (const auto &roi : rois) {
          if (!all_coords(roi.hi >= roi.lo))
            return false;
        }
        return true;
    }(), "One or more regions of interests are invalid");
    DALI_ENFORCE([=]() -> bool {
        auto ref_nchannels = in.shape[0][ndims - 1];
        for (int i = 0; i < in.num_samples(); i++) {
          if (in.shape[i][ndims - 1] != ref_nchannels) {
            return false;
          }
        }
        return true;
    }(), "Number of channels for every image in batch must be equal");

    auto adjusted_rois = AdjustRoi(make_cspan(rois), in.shape);
    auto nchannels = in.shape[0][ndims - 1];
    KernelRequirements req;
    ScratchpadEstimator se;
    auto sh = ShapeFromRoi(make_cspan(adjusted_rois), nchannels);
    TensorListShape<spatial_dims> flattened_shape(multiply_add::FlattenChannels<ndims>(sh));
    block_setup_.SetupBlocks(flattened_shape, true);
    sample_descriptors_.resize(in.num_samples());
    se.add<mm::memory_kind::device, SampleDesc>(in.num_samples());
    se.add<mm::memory_kind::device, BlockDesc>(block_setup_.Blocks().size());
    req.output_shapes = {in.shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(
      KernelContext &context,
      const OutListGPU<OutputType, ndims> &out, const InListGPU<InputType, ndims> &in,
      const std::vector<float> &addends, const std::vector<float> &multipliers,
      const std::vector<Roi<spatial_dims>> &rois = {}) {
    multiply_add::CreateSampleDescriptors(
        make_span(sample_descriptors_), out, in, addends, multipliers);

    SampleDesc *samples_gpu;
    BlockDesc *blocks_gpu;

    std::tie(samples_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
            context.gpu.stream, sample_descriptors_, block_setup_.Blocks());

    dim3 grid_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    auto stream = context.gpu.stream;

    multiply_add::MultiplyAddKernel<<<grid_dim, block_dim, 0, stream>>>(samples_gpu, blocks_gpu);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_POINTWISE_MULTIPLY_ADD_GPU_H_
