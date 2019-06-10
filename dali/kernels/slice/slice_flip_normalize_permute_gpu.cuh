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

#ifndef DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_
#define DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_

#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include "dali/kernels/slice/slice_flip_normalize_permute_common.h"
#include "dali/kernels/kernel.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/dev_array.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {

namespace detail {

template <size_t Dims>
struct SampleDesc {
  void *__restrict__ out;
  const void *__restrict__ in;
  DeviceArray<int64_t, Dims> in_strides;
  DeviceArray<int64_t, Dims> out_strides;
  DeviceArray<int64_t, Dims> out_shape;
  DeviceArray<int64_t, Dims> padded_out_shape;
  size_t normalization_dim;
};

struct BlockDesc {
  int sampleIdx;
  size_t offset;
  size_t size;
};

template <unsigned Dims, typename OutputType, typename InputType>
__device__ void SliceFlipNormalizePermuteFunc(OutputType *__restrict__ out,
                                              const InputType *__restrict__ in,
                                              const int64_t *out_strides,
                                              const int64_t *in_strides,
                                              const int64_t *out_shape,
                                              bool should_normalize,
                                              size_t norm_dim,
                                              const float *mean,
                                              const float *inv_stddev,
                                              size_t offset, size_t block_end) {
  if (Dims > 1 && out_strides[Dims - 1] == in_strides[Dims - 1] && (!should_normalize || norm_dim != (Dims - 1))) {
    const unsigned NextDims = Dims > 1 ? Dims - 1 : 1;
    SliceFlipNormalizePermuteFunc<NextDims>(out, in, out_strides, in_strides, out_shape,
                                            should_normalize, norm_dim, mean, inv_stddev, offset,
                                            block_end);
    return;
  }

  for (; offset < block_end; offset += blockDim.x) {
    size_t idx = offset;
    size_t out_idx = idx;
    size_t in_idx = 0;
    size_t norm_i = 0;
    bool zero_pad = false;
    for (unsigned d = 0; d < Dims; d++) {
      unsigned i_d = idx / static_cast<size_t>(out_strides[d]);
      if ( zero_pad = (i_d >= out_shape[d]) ) {
        break;
      }
      if (should_normalize && d == norm_dim) {
        norm_i = i_d;
      }
      idx %= static_cast<size_t>(out_strides[d]);
      in_idx += i_d * in_strides[d];
    }

    if (zero_pad) {
      out[out_idx] = static_cast<OutputType>(0);
    } else {
      in_idx += idx;  // remaining dims have equal strides
      if (should_normalize) {
        float element = (static_cast<float>(in[in_idx]) - mean[norm_i]) * inv_stddev[norm_i];
        out[out_idx] = clamp<OutputType>(element);
      } else {
        out[out_idx] = clamp<OutputType>(in[in_idx]);
      }
    }
  }
}

template <typename OutputType, typename InputType, size_t Dims>
__global__ void SliceFlipNormalizePermuteKernel(const SampleDesc<Dims> *samples,
                                                const BlockDesc *blocks,
                                                const float *mean,
                                                const float *inv_stddev,
                                                size_t normalization_dim) {
  int sampleIdx = blocks[blockIdx.x].sampleIdx;
  size_t offset = blocks[blockIdx.x].offset + threadIdx.x;
  size_t block_end = blocks[blockIdx.x].offset + blocks[blockIdx.x].size;
  auto sample = samples[sampleIdx];
  auto *out = static_cast<OutputType *>(sample.out);
  auto *in = static_cast<const InputType *>(sample.in);

  // TODO(janton): fill in
  bool should_normalize = mean != nullptr && inv_stddev != nullptr;
  SliceFlipNormalizePermuteFunc<Dims>(out, in, sample.out_strides.data(), sample.in_strides.data(),
                                      sample.out_shape.data(), should_normalize, normalization_dim,
                                      mean, inv_stddev, offset, block_end);
}

}  // namespace detail

template <typename OutputType, typename InputType, size_t Dims>
class SliceFlipNormalizePermuteGPU {
 private:
  static constexpr size_t kBlockDim = 256;
  static constexpr size_t kBlockSize = 64 * kBlockDim;
  size_t block_count_ = 0;

 public:
  using Args = SliceFlipNormalizePermuteArgs<Dims>;
  KernelRequirements Setup(KernelContext &context,
                           const InListGPU<InputType, Dims> &in,
                           const std::vector<Args> &args) {
    KernelRequirements req;
    ScratchpadEstimator se;
    const size_t num_samples = in.size();
    se.add<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    se.add<detail::SampleDesc<Dims>>(AllocType::GPU, num_samples);
    se.add<float>(AllocType::GPU, args[0].mean.size());
    se.add<float>(AllocType::GPU, args[0].inv_stddev.size());

    std::vector<size_t> sample_sizes;
    sample_sizes.reserve(args.size());
    for (auto &elem : args) {
      sample_sizes.push_back(volume(elem.shape));
    }

    block_count_ = 0;
    for (size_t sample_size : sample_sizes) {
      block_count_ += std::ceil(
        sample_size / static_cast<float>(kBlockSize));
    }

    se.add<detail::BlockDesc>(AllocType::Host, block_count_);
    se.add<detail::BlockDesc>(AllocType::GPU, block_count_);
    req.scratch_sizes = se.sizes;

    TensorListShape<Dims> output_shapes;
    auto in_shapes = in.shape;
    output_shapes.resize(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      TensorShape<Dims> out_shape(args[i].padded_shape);
      CheckValidOutputShape<Dims>(in_shapes[i], out_shape, args[i]);
      out_shape = detail::permute<Dims>(out_shape, args[i].permuted_dims);
      output_shapes.set_tensor_shape(i, out_shape);
    }
    req.output_shapes = { output_shapes };
    return req;
  }

  void Run(KernelContext &context,
           OutListGPU<OutputType, Dims> &out,
           const InListGPU<InputType, Dims> &in,
           const std::vector<Args> &args) {
    const auto num_samples = in.size();

    size_t normalization_dim = args[0].normalization_dim;
    auto mean_data = args[0].mean;
    auto inv_stddev_data = args[0].inv_stddev;
    DALI_ENFORCE(mean_data.size() == inv_stddev_data.size());

    detail::SampleDesc<Dims>* sample_descs_cpu =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(AllocType::Host, num_samples);
    float *mean_cpu = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, mean_data.size());
    float *inv_stddev_cpu = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(AllocType::Host, inv_stddev_data.size());
    detail::BlockDesc *block_descs_cpu =
      context.scratchpad->Allocate<detail::BlockDesc>(AllocType::Host, block_count_);

    for (size_t i = 0; i < mean_data.size(); i++) {
      mean_cpu[i] = mean_data[i];
      inv_stddev_cpu[i] = inv_stddev_data[i];
    }
    std::cout << "mean"; for (size_t i = 0; i < mean_data.size(); i++) std::cout << " " << mean_cpu[i]; std::cout << std::endl;
    std::cout << "std"; for (size_t i = 0; i < mean_data.size(); i++) std::cout << " " << inv_stddev_cpu[i]; std::cout << std::endl;

    std::vector<size_t> sample_sizes(in.size());
    for (int i = 0; i < in.size(); i++) {
      const auto in_shape = in.tensor_shape(i);
      auto processed_args = detail::ProcessArgs<Dims>(args[i], in_shape);

      auto &sample_desc = sample_descs_cpu[i];
      sample_desc.in_strides = processed_args.in_strides;
      std::cout << "in strides"; for (auto &x : sample_desc.in_strides) std::cout << " " << x; std::cout << std::endl;
      sample_desc.out_strides = processed_args.out_strides;
      std::cout << "out_strides"; for (auto &x : sample_desc.out_strides) std::cout << " " << x; std::cout << std::endl;
      sample_desc.out_shape = processed_args.out_shape;
      std::cout << "out_shape"; for (auto &x : sample_desc.out_shape) std::cout << " " << x; std::cout << std::endl;
      sample_desc.padded_out_shape = processed_args.padded_out_shape;
      std::cout << "padded_out_shape"; for (auto &x : sample_desc.padded_out_shape) std::cout << " " << x; std::cout << std::endl;
      sample_desc.in = in.tensor_data(i) + processed_args.input_offset;
      std::cout << "input_offset"; std::cout << " " << processed_args.input_offset << std::endl;
      sample_desc.out = out.tensor_data(i);
      sample_sizes[i] = volume(processed_args.padded_out_shape);
    }

    size_t block_idx = 0;
    for (int i = 0; i < num_samples; i++) {
      size_t offset = 0;
      size_t remaining = sample_sizes[i];
      while (remaining > 0) {
        size_t size = remaining < kBlockSize ? remaining : kBlockSize;
        block_descs_cpu[block_idx++] = {i, offset, size};
        remaining -= size;
        offset += size;
      }
    }

    detail::SampleDesc<Dims> *sample_descs =
      context.scratchpad->Allocate<detail::SampleDesc<Dims>>(
        AllocType::GPU, num_samples);

    float *mean = mean_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, mean_data.size());

    float *inv_stddev = inv_stddev_data.empty() ? nullptr :
      context.scratchpad->Allocate<float>(
        AllocType::GPU, inv_stddev_data.size());

    detail::BlockDesc *block_descs =
      context.scratchpad->Allocate<detail::BlockDesc>(
        AllocType::GPU, block_count_);

    // Memory is allocated contiguously, so we launch only one cudaMemcpyAsync
    size_t total_bytes = num_samples * sizeof(detail::SampleDesc<Dims>)
      + mean_data.size() * sizeof(float)
      + inv_stddev_data.size() * sizeof(float)
      + block_count_ * sizeof(detail::BlockDesc);
    cudaMemcpyAsync(sample_descs, sample_descs_cpu,
                    total_bytes,
                    cudaMemcpyHostToDevice,
                    context.gpu.stream);

    const auto grid = block_count_;
    detail::SliceFlipNormalizePermuteKernel<OutputType, InputType, Dims>
      <<<grid, kBlockDim, 0, context.gpu.stream>>>(sample_descs, block_descs, mean, inv_stddev, normalization_dim);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_FLIP_NORMALIZE_PERMUTE_GPU_H_
