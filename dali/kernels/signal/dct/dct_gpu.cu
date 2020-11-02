// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
  int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  auto block = blocks[bid];
  const auto &sample = samples[block.sample_idx];
  ivec3 in_stride = sample.in_stride;
  ivec3 out_stride = sample.out_stride;

  for (int z = block.start.z + threadIdx.z; z < block.end.z; z += blockDim.z) {
    for (int y = block.start.y + threadIdx.y; y < block.end.y; y += blockDim.y) {
      const OutputType *cos_row = sample.cos_table + sample.input_length * y;
      float coeff = HasLifter ? lifter_coeffs[y] : 1.f;
      for (int x = block.start.x + threadIdx.x; x < block.end.x; x += blockDim.x) {
        int output_idx = dot(out_stride, ivec3{z, y, x});
        const InputType *input = sample.input + dot(in_stride, ivec3{z, 0, x});
        OutputType out_val = 0;
        for (int i = 0; i < sample.input_length; ++i) {
          out_val += *input * cos_row[i];
          input += in_stride[1];
        }
        sample.output[output_idx] = HasLifter ? out_val * coeff : out_val;
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
  ScratchpadEstimator se{};
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
      se.add<OutputType>(AllocType::GPU, n * arg.ndct);
      if (n * arg.ndct > max_cos_table_size_) {
        max_cos_table_size_ = n * arg.ndct;
      }
    }
    reduced_shape.set_tensor_shape(s, reduce_shape(in_shape, axis_, arg.ndct));
    auto sample_shape = in.shape[s];
    sample_shape[axis_] = arg.ndct;
    out_shape.set_tensor_shape(s, sample_shape);
  }
  se.add<OutputType>(AllocType::Pinned, max_cos_table_size_);
  if (cos_tables_.size() > 1) {
    se.add<OutputType>(AllocType::Pinned, max_cos_table_size_);
  }
  se.add<SampleDesc>(AllocType::GPU, in.num_samples());
  block_setup_.SetupBlocks(reduced_shape, true);
  se.add<BlockDesc<3>>(AllocType::GPU, block_setup_.Blocks().size());
  req.output_shapes = {out_shape};
  req.scratch_sizes = se.sizes;
  return req;
}

template <typename OutputType, typename InputType>
DLL_PUBLIC void Dct1DGpu<OutputType, InputType>::Run(KernelContext &ctx,
                                                     const OutListGPU<OutputType> &out,
                                                     const InListGPU<InputType> &in,
                                                     InTensorGPU<float, 1> lifter_coeffs) {
  OutputType *cpu_cos_table[2];
  cpu_cos_table[0] =
    ctx.scratchpad->Allocate<OutputType>(AllocType::Pinned, max_cos_table_size_);
  if (cos_tables_.size() > 1) {
    cpu_cos_table[1] =
      ctx.scratchpad->Allocate<OutputType>(AllocType::Pinned, max_cos_table_size_);
  }
  int i = 0;
  for (auto &table_entry : cos_tables_) {
    auto cpu_table = cpu_cos_table[i % 2];
    auto &buffer_event = buffer_events_[i % 2];
    int n;
    DctArgs arg;
    std::tie(n, arg) = table_entry.first;
    cudaEventSynchronize(buffer_event);
    FillCosineTable(cpu_table, n, arg);
    table_entry.second = ctx.scratchpad->ToGPU(ctx.gpu.stream,
                                               span<OutputType>(cpu_table, n * arg.ndct));
    cudaEventRecord(buffer_event, ctx.gpu.stream);
    ++i;
  }
  sample_descs_.clear();
  sample_descs_.reserve(args_.size());
  int s = 0;
  for (auto arg : args_) {
    auto in_shape = reduce_shape(in.tensor_shape_span(s), axis_);
    auto out_shape = reduce_shape(out.tensor_shape_span(s), axis_);
    DALI_ENFORCE(lifter_coeffs.num_elements() == 0 || out_shape[1] <= lifter_coeffs.num_elements(),
                 make_string("Not enough lifter coefficients. NDCT for sample ", s, " is ",
                             out_shape[1], " and only ", lifter_coeffs.num_elements(),
                             " coefficients were passed."));
    ivec3 out_stride = GetStrides(ivec3{out_shape[0], out_shape[1], out_shape[2]});
    ivec3 in_stride = GetStrides(ivec3{in_shape[0], in_shape[1], in_shape[2]});;
    int n = in_shape[1];
    auto *cos_table = cos_tables_[{n, arg}];
    sample_descs_.push_back(SampleDesc{out.tensor_data(s), in.tensor_data(s),
                                       cos_table, in_stride, out_stride, n});
    ++s;
  }
  SampleDesc *sample_descs_gpu;
  BlockDesc<3> *block_descs_gpu;
  std::tie(sample_descs_gpu, block_descs_gpu) =
    ctx.scratchpad->ToContiguousGPU(ctx.gpu.stream, sample_descs_, block_setup_.Blocks());
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  if (lifter_coeffs.num_elements() > 0) {
    ApplyDct<OutputType, InputType, true>
      <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                   lifter_coeffs.data);
  } else {
    ApplyDct<OutputType, InputType, false>
      <<<grid_dim, block_dim, 0, ctx.gpu.stream>>>(sample_descs_gpu, block_descs_gpu,
                                                   nullptr);
  }
}

template class Dct1DGpu<float, float>;

template class Dct1DGpu<double, double>;

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
