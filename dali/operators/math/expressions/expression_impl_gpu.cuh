// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
#define DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_

#include <vector>

#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_factory.h"
#include "dali/operators/math/expressions/expression_tree.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/fast_div.h"

namespace dali {
namespace expr {

// Use BinaryArithmeticOpGpuPerfTest for tuning
static constexpr int kThreadNum = 256;
static constexpr int kBlocksX = 64;

template <int nargs>
struct SampleDescGPU {
  struct {
    void *data;
    DALIDataType dtype;
    fast_div<uint64_t> strides[ARITHM_OPS_MAX_DIM];
    int64_t shape[ARITHM_OPS_MAX_DIM];  // NOLINT[runtime/arrays]
  } output;

  struct {
    const void *data;
    DALIDataType dtype;
    int64_t shape[ARITHM_OPS_MAX_DIM];  // NOLINT[runtime/arrays]
    int64_t strides[ARITHM_OPS_MAX_DIM];  // NOLINT[runtime/arrays]
  } args[nargs];

  int ndim;
};

template <int NumArgs>
bool CanUseFlatIdx(span<SampleDescGPU<NumArgs>> samples) {
  for (int i = 0; i < samples.size(); i++) {
    if (samples[i].ndim > 1)
      return false;
    for (int a = 0; a < NumArgs; a++) {
      if (samples[i].args[a].strides[0] > 1)
        return false;
    }
  }
  return true;
}

inline dim3 GetGridLayout(int extent, int tiles) {
  return dim3(extent, tiles, 1);
}

template <typename Invoker, int NumArgs>
void ExecuteImpl(ExprImplContext &ctx, span<const SampleDesc> samples,
                 span<const TileDesc> tiles) {
  kernels::DynamicScratchpad s(ctx.stream);

  assert(samples.size() > 0);
  int ndim = samples[0].output.shape.sample_dim();

  assert(ndim < ARITHM_OPS_MAX_DIM);  // should be checked earlier
  for (int i = 0; i < samples.size(); i++) {
    assert(ndim == samples[i].output.shape.sample_dim());
    assert(NumArgs == samples[i].args.size());
  }

  auto samples_cpu =
      make_span(s.Allocate<mm::memory_kind::host, SampleDescGPU<NumArgs>>(samples.size()),
                samples.size());
  FillSampleDesc(samples_cpu, samples);
  bool can_use_flat_idx = CanUseFlatIdx(samples_cpu);

  SampleDescGPU<NumArgs>* samples_gpu;
  TileDesc *tiles_gpu;
  std::tie(samples_gpu, tiles_gpu) = s.ToContiguousGPU(ctx.stream, samples_cpu, tiles);
  auto grid = GetGridLayout(kBlocksX, tiles.size());
  auto block = dim3(kThreadNum, 1, 1);
  Invoker::Invoke(samples_gpu, tiles_gpu, grid, block, ctx.stream, can_use_flat_idx);
}

template <int nargs>
void FillSampleDesc(span<SampleDescGPU<nargs>> sample_descs, span<const SampleDesc> samples) {
  assert(sample_descs.size() == samples.size());
  for (int i = 0; i < samples.size(); i++) {
    auto &sample_desc = sample_descs[i];
    auto &sample = samples[i];
    sample_desc.ndim = sample.output.shape.sample_dim();
    sample_desc.output.data = sample.output.data;
    sample_desc.output.dtype = sample.output.dtype;
    for (int d = 0; d < sample_desc.ndim; d++) {
      sample_desc.output.shape[d] = sample.output.shape[d];
      sample_desc.output.strides[d] = sample.output.strides[d];
    }

    for (int operand_idx = 0; operand_idx < nargs; operand_idx++) {
      auto &operand_desc = sample_desc.args[operand_idx];
      auto &operand = sample.args[operand_idx];
      operand_desc.data = operand.data;
      operand_desc.dtype = operand.dtype;
      assert(sample_desc.ndim == operand.shape.sample_dim());
      for (int d = 0; d < sample_desc.ndim; d++) {
        operand_desc.shape[d] = operand.shape[d];
        operand_desc.strides[d] = operand.strides[d];
      }
    }
  }
}

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
