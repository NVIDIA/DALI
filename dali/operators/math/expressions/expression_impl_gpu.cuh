// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <int nargs>
struct SampleDesc1DGPU {
  struct {
    void *data;
    DALIDataType dtype;
    int64_t extent;
  } output;

  struct {
    const void *data;
    DALIDataType dtype;
    int64_t extent;
  } args[nargs];
};

template <int nargs, int ndim>
void FillSampleDesc(span<SampleDescGPU<nargs, ndim>> sample_descs, span<const SampleDesc> samples) {
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

inline dim3 GetGridLayout(int extent, int tiles) {
  return dim3(extent, tiles, 1);
}
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_EXPRESSION_IMPL_GPU_CUH_
