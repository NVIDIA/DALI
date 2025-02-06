// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/dct/dct_gpu.h"
#include <cmath>
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/dct/table.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

// The kernel processes data with the shape reduced to 3D.
// Transform is applied over the middle axis.
template <typename OutputType, typename InputType, bool HasLifter>
__global__ void ApplyDct(const typename Dct1DGpu<OutputType, InputType>::SampleDesc *samples,
                         const BlockDesc<3> *blocks,  const float *lifter_coeffs)  {
  extern __shared__ char cos_table_shm[];
  OutputType *cos_table = reinterpret_cast<OutputType*>(cos_table_shm);
  int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_vol = blockDim.x * blockDim.y * blockDim.z;
  auto block = blocks[bid];
  const auto &sample = samples[block.sample_idx];
  ivec3 in_stride = sample.in_stride;
  ivec3 out_stride = sample.out_stride;
  const OutputType *cos_table_inp = sample.cos_table + block.start.y * sample.input_length;
  size_t size = (block.end.y - block.start.y) * sample.input_length;
  for (size_t i = tid; i < size; i += block_vol) {
    cos_table[i] = cos_table_inp[i];
  }
  __syncthreads();
  for (int z = block.start.z + threadIdx.z; z < block.end.z; z += blockDim.z) {
    for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
      const OutputType *cos_row = cos_table + sample.input_length * (y - block.start.y);
      float coeff = HasLifter ? lifter_coeffs[y] : 1.f;
      for (int x = block.start.x + threadIdx.x; x < block.end.x; x += blockDim.x) {
        int output_idx = out_stride[0]*z + out_stride[1]*y + x;
        const InputType *input = sample.input + in_stride[0]*z + x;
        OutputType out_val = 0;
        for (int i = 0; i < sample.input_length; ++i) {
          out_val = fma(*input, cos_row[i], out_val);
          input += in_stride[1];
        }
        sample.output[output_idx] = HasLifter ? out_val * coeff : out_val;
      }
    }
  }
}

template <typename OutputType, typename InputType, bool HasLifter>
__global__ void ApplyDctInner(const typename Dct1DGpu<OutputType, InputType>::SampleDesc *samples,
                              const BlockSetupInner::BlockDesc *blocks,
                              const float *lifter_coeffs) {
  extern __shared__ char shm[];
  auto block = blocks[blockIdx.x];
  auto sample = samples[block.sample_idx];
  int ndct = sample.out_stride[0];
  int64_t nframes = block.frame_count;
  int input_len = sample.input_length * nframes;
  auto *in_frames = reinterpret_cast<InputType*>(shm);
  auto *cos_table = reinterpret_cast<OutputType*>(in_frames + input_len);
  auto *input = sample.input + block.frame_start * sample.input_length;
  auto *output = sample.output + block.frame_start * ndct;
  int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_vol = blockDim.x * blockDim.y * blockDim.z;
  int table_len = sample.input_length * ndct;
  for (int idx = tid; idx < table_len; idx += block_vol) {
    cos_table[idx] = sample.cos_table[idx];
  }
  for (int idx = tid; idx < input_len; idx += block_vol) {
    in_frames[idx] = input[idx];
  }
  __syncthreads();
  for (int f = 0; f < nframes; ++f) {
    const auto *in_frame = in_frames + sample.input_length * f;
    auto *out_frame = output + ndct * f;
    for (int y = threadIdx.y; y < ndct; y += blockDim.y) {
      float lifter_coeff = HasLifter ? lifter_coeffs[y] : 1.f;
      const auto *cos_row = &cos_table[y * sample.input_length];
      OutputType acc = 0;
      for (int x = threadIdx.x; x < sample.input_length; x += blockDim.x) {
        acc = fma(in_frame[x], cos_row[x], acc);
      }
      acc += __shfl_down_sync(0xffffffff, acc, 16);
      acc += __shfl_down_sync(0xffffffff, acc, 8);
      acc += __shfl_down_sync(0xffffffff, acc, 4);
      acc += __shfl_down_sync(0xffffffff, acc, 2);
      acc += __shfl_down_sync(0xffffffff, acc, 1);
      if (threadIdx.x == 0) {
        out_frame[y] = HasLifter ? acc * lifter_coeff : acc;
      }
    }
  }
}

template <typename OutputType, typename InputType>
KernelRequirements Dct1DGpu<OutputType, InputType>::Setup(KernelContext &ctx,
                                                          const InListGPU<InputType> &in,
                                                          span<const DctArgs> args, int axis) {
  DALI_ENFORCE(args.size() == in.num_samples());
  KernelRequirements req{};
  args_.clear();
  cos_tables_.clear();
  sample_descs_.clear();
  int64_t dims = in.sample_dim();
  TensorListShape<> out_shape(in.num_samples(), dims);
  TensorListShape<3> reduced_shape(in.num_samples());
  max_cos_table_size_ = 0;
  axis_ = axis >= 0 ? axis : dims - 1;
  DALI_ENFORCE(axis_ >= 0 && axis_ < dims,
               make_string("Axis is out of bounds: ", axis_));
  inner_axis_ = true;
  for (int s = 0; s < args.size(); ++s) {
    args_.push_back(args[s]);
    auto &arg = args_.back();
    auto in_shape = in.tensor_shape_span(s);
    int64_t n = in_shape[axis_];

    if (arg.dct_type == 1) {
      DALI_ENFORCE(n > 1, "DCT type I requires an input length > 1");
      if (arg.normalize) {
        DALI_WARN("DCT type-I does not support orthogonal normalization. Ignoring");
        arg.normalize = false;
      }
    }

    if (arg.ndct <= 0) {
      arg.ndct = n;
    }
    if (cos_tables_.find({n, arg}) == cos_tables_.end()) {
      cos_tables_[{n, arg}] = nullptr;
      if (n * arg.ndct > max_cos_table_size_) {
        max_cos_table_size_ = n * arg.ndct;
      }
    }
    auto reduced_samle_shape = reduce_shape(in_shape, axis_, arg.ndct);
    reduced_shape.set_tensor_shape(s, reduced_samle_shape);
    if (reduced_samle_shape[2] != 1)
      inner_axis_ = false;
    auto sample_shape = in.shape[s];
    sample_shape[axis_] = arg.ndct;
    out_shape.set_tensor_shape(s, sample_shape);
  }
  if (inner_axis_) {
    block_setup_inner_.Setup(reduced_shape);
  } else {
    block_setup_.SetupBlocks(reduced_shape, true);
  }
  req.output_shapes = {out_shape};
  return req;
}

template <typename OutputType, typename InputType>
DLL_PUBLIC void Dct1DGpu<OutputType, InputType>::Run(KernelContext &ctx,
                                                     const OutListGPU<OutputType> &out,
                                                     const InListGPU<InputType> &in,
                                                     InTensorGPU<float, 1> lifter_coeffs) {
  OutputType *cpu_cos_table[2];
  cpu_cos_table[0] = ctx.scratchpad->AllocatePinned<OutputType>(max_cos_table_size_);
  if (cos_tables_.size() > 1) {
    cpu_cos_table[1] = ctx.scratchpad->AllocatePinned<OutputType>(max_cos_table_size_);
  }

  int i = 0;
  for (auto &table_entry : cos_tables_) {
    auto cpu_table = cpu_cos_table[i % 2];
    auto &buffer_event = buffer_events_[i % 2];
    int n;
    DctArgs arg;
    std::tie(n, arg) = table_entry.first;
    CUDA_CALL(cudaEventSynchronize(buffer_event));
    FillCosineTable(cpu_table, n, arg);
    table_entry.second = ctx.scratchpad->ToGPU(ctx.gpu.stream,
                                               span<OutputType>(cpu_table, n * arg.ndct));
    CUDA_CALL(cudaEventRecord(buffer_event, ctx.gpu.stream));
    ++i;
  }
  sample_descs_.clear();
  sample_descs_.reserve(args_.size());
  int s = 0;
  int max_ndct = 0;
  int max_input_length = 0;
  for (const auto &arg : args_) {
    auto in_shape = reduce_shape(in.tensor_shape_span(s), axis_);
    auto out_shape = reduce_shape(out.tensor_shape_span(s), axis_);
    DALI_ENFORCE(lifter_coeffs.num_elements() == 0 || out_shape[1] <= lifter_coeffs.num_elements(),
                 make_string("Not enough lifter coefficients. NDCT for sample ", s, " is ",
                             out_shape[1], " and only ", lifter_coeffs.num_elements(),
                             " coefficients were passed."));
    ivec3 out_stride = GetStrides(ivec3{out_shape[0], out_shape[1], out_shape[2]});
    ivec3 in_stride = GetStrides(ivec3{in_shape[0], in_shape[1], in_shape[2]});;
    int n = in_shape[1];
    auto *cos_tables = cos_tables_[{n, arg}];
    sample_descs_.push_back(SampleDesc{out.tensor_data(s), in.tensor_data(s),
                                       cos_tables, in_stride, out_stride, n});
    max_ndct = std::max(max_ndct, arg.ndct);
    max_input_length = std::max(max_input_length, n);
    ++s;
  }
  if (inner_axis_) {
    RunInnerDCT(ctx, max_input_length, lifter_coeffs);
  } else {
    RunPlanarDCT(ctx, max_ndct, lifter_coeffs);
  }
}

void BlockSetupInner::Setup(const TensorListShape<3> &reduced_shape) {
  blocks_.clear();
  int64_t bid = 0;
  for (int s = 0; s < reduced_shape.num_samples(); ++s) {
    assert(reduced_shape[s][2] == 1);
    int64_t nframes = reduced_shape[s][0];
    int64_t nblocks = div_ceil(nframes, frames_per_block_);
    blocks_.resize(blocks_.size() + nblocks);
    for (int64_t f = 0; f < nframes; f += frames_per_block_, ++bid) {
      blocks_[bid].sample_idx = s;
      blocks_[bid].frame_start = f;
      blocks_[bid].frame_count = std::min(frames_per_block_, nframes - f);
    }
  }
}

template <typename OutputType, typename InputType>
void Dct1DGpu<OutputType, InputType>::RunInnerDCT(KernelContext &ctx, int64_t max_input_length,
                                                  InTensorGPU<float, 1> lifter_coeffs) {
  SampleDesc *sample_descs_gpu;
  BlockSetupInner::BlockDesc *block_descs_gpu;
  std::tie(sample_descs_gpu, block_descs_gpu) =
    ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_, block_setup_inner_.Blocks());
  dim3 block_dim = block_setup_inner_.BlockDim();
  dim3 grid_dim = block_setup_inner_.GridDim();
  size_t shm_size =
    block_setup_inner_.SharedMemSize<OutputType, InputType>(max_input_length, max_cos_table_size_);
  if (lifter_coeffs.num_elements() > 0) {
    ApplyDctInner<OutputType, InputType, true>
      <<<grid_dim, block_dim, shm_size, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                          lifter_coeffs.data);
  } else {
    ApplyDctInner<OutputType, InputType, false>
      <<<grid_dim, block_dim, shm_size, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                          nullptr);
  }
}

template <typename OutputType, typename InputType>
void Dct1DGpu<OutputType, InputType>::RunPlanarDCT(KernelContext &ctx, int max_ndct,
                                                   InTensorGPU<float, 1> lifter_coeffs) {
  SampleDesc *sample_descs_gpu;
  BlockDesc<3> *block_descs_gpu;
  std::tie(sample_descs_gpu, block_descs_gpu) =
    ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_, block_setup_.Blocks());
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  size_t shm_size = sizeof(OutputType) * (max_cos_table_size_ + 32 * max_ndct);
  auto block = block_setup_.Blocks()[0];
  if (lifter_coeffs.num_elements() > 0) {
    ApplyDct<OutputType, InputType, true>
      <<<grid_dim, block_dim, shm_size, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                          lifter_coeffs.data);
  } else {
    ApplyDct<OutputType, InputType, false>
      <<<grid_dim, block_dim, shm_size, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                          nullptr);
  }
}

template class Dct1DGpu<float, float>;

template class Dct1DGpu<double, double>;

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
