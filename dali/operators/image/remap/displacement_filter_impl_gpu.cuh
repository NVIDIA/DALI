// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
#define DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_GPU_CUH_

#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/host_dev.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/sampler.h"
#include "dali/operators/image/remap/displacement_filter.h"
namespace dali {

struct DisplacementSampleDesc {
  void *output;
  const void *input;
  const void *raw_params;
  TensorShape<3> shape;
  bool mask;
};

template <typename T, class Displacement, DALIInterpType interp_type>
__device__ inline T GetPixelValueSingleC(int h, int w, int c,
                                         int H, int W, int C,
                                         const T * input,
                                         Displacement& displace, const T fill_value) {
  kernels::Surface2D<const T> in_surface = { input, W, H, C, C, C*W, 1 };
  auto sampler = kernels::make_sampler<interp_type>(in_surface);
  auto p = displace(h, w, c, H, W, C);
  return sampler.template at<T>(p, c, fill_value);
}

template <typename T, class Displacement, DALIInterpType interp_type>
__device__ inline void GetPixelValueMultiC(int h, int w,
                                           int H, int W, int C,
                                           const T * input, T * output,
                                           Displacement& displace, const T fill_value) {
  kernels::Surface2D<const T> in_surface = { input, W, H, C, C, C*W, 1 };
  auto sampler = kernels::make_sampler<interp_type>(in_surface);
  auto p = displace(h, w, 0, H, W, C);
  sampler(output, p, fill_value);
}

template <class Displacement, bool has_param = HasParam<Displacement>::value>
struct PrepareParam {
  __device__ __host__ inline void operator() (Displacement &, const void *) {}
};

template <class Displacement>
struct PrepareParam<Displacement, true> {
  __device__ __host__ inline void operator()(Displacement &displace, const void *raw_params) {
    const auto *const params = static_cast<const typename Displacement::Param *>(raw_params);
    displace.param = *params;
  }
};

template <typename T, bool per_channel_transform, class Displacement, DALIInterpType interp_type>
__global__ void DisplacementKernel(const DisplacementSampleDesc *samples,
                                   const kernels::BlockDesc<1> *blocks, const T fill_value,
                                   Displacement displace) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  auto *image_out = static_cast<T *>(sample.output);
  const auto *image_in = static_cast<const T *>(sample.input);

  const int H = sample.shape[0];
  const int W = sample.shape[1];
  const int C = sample.shape[2];
  const bool m = sample.mask;

  PrepareParam<Displacement> pp;
  pp(displace, sample.raw_params);

  auto start = block.start.x;
  auto end = block.end.x;
  for (int64_t out_idx = threadIdx.x + start; out_idx < end; out_idx += blockDim.x) {
    if (m) {
      int64_t idx = out_idx;
      const int c = idx % C;
      idx /= C;
      const int w = idx % W;
      idx /= W;
      const int h = idx;

      image_out[out_idx] = GetPixelValueSingleC<T, Displacement, interp_type>(
          h, w, c, H, W, C, image_in, displace, fill_value);
    } else {
      image_out[out_idx] = image_in[out_idx];
    }
  }
}

template <typename T, int C, bool per_channel_transform,
          int nThreads, class Displacement, DALIInterpType interp_type>
__global__
void DisplacementKernel_aligned32bit(
  const DisplacementSampleDesc *samples,
                                  const kernels::BlockDesc<1> *blocks,
                        const T fill_value,
                        Displacement displace) {
  constexpr int nPixelsPerThread = sizeof(uint32_t)/sizeof(T);
  __shared__ T scratch[nThreads * C * nPixelsPerThread];

  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];

  auto *image_out = reinterpret_cast<uint32_t *>(sample.output);
  const auto *image_in = reinterpret_cast<const T *>(sample.input);

  const int H = sample.shape[0];
  const int W = sample.shape[1];
  const bool m = sample.mask;

  PrepareParam<Displacement> pp;
  pp(displace, sample.raw_params);

  auto start = block.start.x;
  auto start_pixelgroup = start / nPixelsPerThread;
  auto end = block.end.x;
  if (m) {
    // aligned u32 elements to process
    auto nElements = (end - start) / nPixelsPerThread;
    auto loopCount = nElements / nThreads;
    T *const my_scratch = scratch + threadIdx.x * C * nPixelsPerThread;
    uint32_t *const scratch_32 = reinterpret_cast<uint32_t *>(scratch);

    for (int lidx = 0; lidx < loopCount; ++lidx) {
      const auto hw0 = start + (lidx * nThreads + threadIdx.x) * nPixelsPerThread;
      uint32_t *const current_image_out = image_out + start_pixelgroup * C + lidx * nThreads * C;
      if (per_channel_transform) {
#pragma unroll
        for (int j = 0; j < nPixelsPerThread; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
#pragma unroll
          for (int c = 0; c < C; ++c) {
            my_scratch[j * C + c] = GetPixelValueSingleC<T, Displacement, interp_type>(
                h, w, c, H, W, C, image_in, displace, fill_value);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < nPixelsPerThread; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
          GetPixelValueMultiC<T, Displacement, interp_type>(
              h, w, H, W, C, image_in, my_scratch + j * C, displace, fill_value);
        }
      }
      __syncthreads();

#pragma unroll
      for (int i = 0; i < C; ++i) {
        current_image_out[threadIdx.x + i * nThreads] = scratch_32[threadIdx.x + i * nThreads];
      }
    }

    // The rest, that was not aligned to block boundary
    const int myId = threadIdx.x + loopCount * nThreads;
    if (myId < nElements) {
      auto *out = reinterpret_cast<T *>(sample.output);
      const int hw0 = start + myId * nPixelsPerThread;
      if (per_channel_transform) {
#pragma unroll
        for (int j = 0; j < nPixelsPerThread; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
#pragma unroll
          for (int c = 0; c < C; ++c) {
            out[h * W * C + w * C + c] =
                GetPixelValueSingleC<T, Displacement, interp_type>(h, w, c, H, W, C, image_in,
                                                                   displace, fill_value);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < nPixelsPerThread; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
          GetPixelValueMultiC<T, Displacement, interp_type>(
              h, w, H, W, C, image_in, out + h * W * C + w * C, displace, fill_value);
        }
      }
    }
  } else {
    auto *image_out = reinterpret_cast<uint32_t *>(sample.output);
    const auto *image_in = reinterpret_cast<const uint32_t *>(sample.input);
    auto start_offset = start * C / nPixelsPerThread + threadIdx.x;
    auto end_offset = end * C / nPixelsPerThread;
    for (int64_t i = start_offset; i < end_offset; i += nThreads) {
      image_out[i] = image_in[i];
    }
  }
}

template <class Displacement, bool per_channel_transform>
class DisplacementFilter<GPUBackend, Displacement, per_channel_transform>
    : public DisplacementBase<GPUBackend, Displacement> {
 public:
  explicit DisplacementFilter(const OpSpec &spec) :
      DisplacementBase<GPUBackend, Displacement>(spec),
      displace_(spec),
      interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    channel_block_setup_.SetBlockDim(ivec3{kAlignedBlockDim, 1, 1});
    has_mask_ = spec.HasTensorArgument("mask");
    DALI_ENFORCE(interp_type_ == DALI_INTERP_NN || interp_type_ == DALI_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for this operation");

    if (!spec.TryGetArgument<float>(fill_value_, "fill_value")) {
      int int_value = 0;
      if (!spec.TryGetArgument<int>(int_value, "fill_value")) {
        DALI_FAIL("Invalid type of argument \"fill_value\". Expected int or float");
      }
      fill_value_ = int_value;
    }
  }

  virtual ~DisplacementFilter() {
     displace_.Cleanup();
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    output_desc.resize(1);
    output_desc[0].shape = input.shape();
    output_desc[0].type = input.type();
    return true;
  }

  void RunImpl(Workspace &ws) override {
    PrepareDisplacement(ws);
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    if (IsType<float>(input.type())) {
      BatchedGPUKernel<float>(ws);
    } else if (IsType<uint8_t>(input.type())) {
      BatchedGPUKernel<uint8_t>(ws);
    } else {
      DALI_FAIL(make_string("Unexpected input type ", input.type()));
    }
  }

  template <typename U = Displacement>
  std::enable_if_t<HasParam<U>::value> PrepareDisplacement(Workspace &ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    params_.Resize({curr_batch_size});
    params_.mutable_data<typename U::Param>();

    for (int i = 0; i < curr_batch_size; ++i) {
      auto &p = params_.mutable_data<typename U::Param>()[i];
      displace_.Prepare(&p, spec_, ws, i);
    }
    params_gpu_.Resize(params_.shape());
    params_gpu_.Copy(params_, ws.stream());
  }

  template <typename U = Displacement>
  std::enable_if_t<!HasParam<U>::value> PrepareDisplacement(Workspace &) {}

  template <typename U = Displacement>
  std::enable_if_t<HasParam<U>::value, const typename U::Param *> GetDisplacementParams(
      int sample_idx) {
    return params_.data<typename U::Param>()[sample_idx];
  }

  template <typename U = Displacement>
  std::enable_if_t<!HasParam<U>::value, const void *> GetDisplacementParams(int sample_idx) {
    return nullptr;
  }

  bool ShouldRunAligned(size_t sizeof_T, uint64_t maxPower2, int C) {
    return (maxPower2 >= sizeof(uint32_t) / sizeof_T) && (C == 3 || C == 1);
  }

  USE_OPERATOR_MEMBERS();
  using Operator<GPUBackend>::RunImpl;

 protected:
  Displacement displace_;

 private:
  static const size_t nDims = 3;

  template <typename T>
  bool BatchedGPUKernel(Workspace &ws) {
    const auto &input = ws.Input<GPUBackend>(0);
    const auto &shape = input.shape();
    auto &output = ws.Output<GPUBackend>(0);
    auto stream = ws.stream();

    const auto num_samples = shape.num_samples();
    samples_.resize(num_samples);
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto &sample = samples_[sample_idx];
      sample.output = output.template mutable_tensor<T>(sample_idx);
      sample.input = input.template tensor<T>(sample_idx);
      sample.raw_params = GetDisplacementParams(sample_idx);
      sample.shape = shape.tensor_shape<nDims>(sample_idx);
      sample.mask = has_mask_ ? ws.ArgumentInput("mask").tensor<int>(sample_idx)[0] : true;
    }

    samples_dev_.from_host(samples_, stream);

    // Find if C is the same for all images and
    // what is the maximum power of 2 dividing
    // all H*W
    int C = shape.tensor_shape_span(0)[2];  // First element
    uint64_t maxPower2 = (uint64_t)-1;
    for (int64_t i = 0; i < num_samples; ++i) {
      if (C != shape.tensor_shape_span(i)[2]) {
        C = -1;  // Not all C are the same
      }
      uint64_t HW = 1;
      for (size_t j = 0; j < nDims - 1; ++j) {
        HW *= shape.tensor_shape_span(i)[j];
      }
      uint64_t power2 = HW & (-HW);
      maxPower2 = maxPower2 > power2 ? power2 : maxPower2;
    }

    switch (interp_type_) {
      case DALI_INTERP_NN:
        DisplacementKernelLauncher<T, DALI_INTERP_NN>(shape, C, maxPower2, stream);
        break;
      case DALI_INTERP_LINEAR:
        DisplacementKernelLauncher<T, DALI_INTERP_LINEAR>(shape, C, maxPower2, stream);
        break;
      default:
        DALI_FAIL("Unsupported interpolation type,"
            " only NN and LINEAR are supported for this operation");
    }

    return true;
  }

  template <typename T, DALIInterpType interp_type>
  void DisplacementKernelLauncher(const TensorListShape<> &shape, int C, uint64_t maxPower2,
                                  cudaStream_t stream) {
    if (ShouldRunAligned(sizeof(T), maxPower2, C)) {
      auto collapsed_shape = collapse_dims<2>(shape, {std::make_pair(0, 2)});
      channel_block_setup_.SetupBlocks(collapsed_shape, true);
      blocks_dev_.from_host(channel_block_setup_.Blocks(), stream);
      dim3 grid_dim = channel_block_setup_.GridDim();
      dim3 block_dim = channel_block_setup_.BlockDim();
      switch (C) {
        case 1:
          DisplacementKernel_aligned32bit<T, 1, per_channel_transform, kAlignedBlockDim,
                                          Displacement, interp_type>
              <<<grid_dim, block_dim, 0, stream>>>(samples_dev_.data(), blocks_dev_.data(),
                                                   fill_value_, displace_);
          return;
        case 3:
          DisplacementKernel_aligned32bit<T, 3, per_channel_transform, kAlignedBlockDim,
                                          Displacement, interp_type>
              <<<grid_dim, block_dim, 0, stream>>>(samples_dev_.data(), blocks_dev_.data(),
                                                   fill_value_, displace_);
          return;
        default:
          break;
      }
    }

    auto collapsed_shape = collapse_dims<1>(shape, {std::make_pair(0, shape.sample_dim())});

    flat_block_setup_.SetupBlocks(collapsed_shape, true);
    blocks_dev_.from_host(flat_block_setup_.Blocks(), stream);
    dim3 grid_dim = flat_block_setup_.GridDim();
    dim3 block_dim = flat_block_setup_.BlockDim();
    DisplacementKernel<T, per_channel_transform, Displacement, interp_type>
        <<<grid_dim, block_dim, 0, stream>>>(samples_dev_.data(), blocks_dev_.data(), fill_value_,
                                             displace_);
  }

  DALIInterpType interp_type_;
  float fill_value_;

  // In theory this should be a proper BlockSetup<2, 2> with 2 data dims and channels at the end,
  // but we are keeping the flat addressing for baseline and the flattened variant with channels for
  // the aligned kernel.
  using FlatBlockSetup = kernels::BlockSetup<1, -1>;
  using ChannelBlockSetup = kernels::BlockSetup<1, 1>;

  static constexpr int kAlignedBlockDim = 256;

  FlatBlockSetup flat_block_setup_{32};
  ChannelBlockSetup channel_block_setup_{32};
  std::vector<DisplacementSampleDesc> samples_;

  DeviceBuffer<kernels::BlockDesc<1>> blocks_dev_;
  DeviceBuffer<DisplacementSampleDesc> samples_dev_;

  Tensor<CPUBackend> meta_cpu_;
  Tensor<GPUBackend> meta_gpu_;

  bool has_mask_;
  TensorList<GPUBackend> mask_gpu_;

  Tensor<CPUBackend> params_;
  Tensor<GPUBackend> params_gpu_;
};
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
