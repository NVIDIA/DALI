// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include "dali/core/convert.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/operators/generic/cast.h"

namespace dali {


template <typename OType, typename IType>
__global__ void BatchedCastKernel(const CastSampleDesc *samples,
                                  const kernels::BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  auto *out = reinterpret_cast<OType *>(sample.output);
  const auto *in = reinterpret_cast<const IType *>(sample.input);
  for (int x = threadIdx.x + block.start.x; x < block.end.x; x += blockDim.x) {
    out[x] = ConvertSat<OType>(in[x]);
  }
}


template <>
void Cast<GPUBackend>::PrepareBlocks(const DeviceWorkspace &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  const auto &input_shape = input.shape();
  std::array<std::pair<int, int>, 1> collapse_groups = {{{0, input_shape.sample_dim()}}};
  auto collapsed_shape = collapse_dims<1>(input.shape(), collapse_groups);

  block_setup_.SetupBlocks(collapsed_shape, true);
  blocks_dev_.from_host(block_setup_.Blocks(), ws.stream());
}


template <>
void Cast<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  const auto &input_shape = input.shape();
  auto &output = ws.OutputRef<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  auto num_samples = input_shape.num_samples();
  samples_.resize(num_samples);
  for (int sample_id = 0; sample_id < num_samples; sample_id++) {
    samples_[sample_id].output = output.raw_mutable_tensor(sample_id);
    samples_[sample_id].input = input.raw_tensor(sample_id);
  }
  samples_dev_.from_host(samples_, ws.stream());

  DALIDataType itype = input.type().id();
  dim3 grid_dim = block_setup_.GridDim();
  dim3 block_dim = block_setup_.BlockDim();
  TYPE_SWITCH(output_type_, type2id, OType, CAST_ALLOWED_TYPES, (
    TYPE_SWITCH(itype, type2id, IType, CAST_ALLOWED_TYPES, (
      BatchedCastKernel<OType, IType>
          <<<grid_dim, block_dim, 0, ws.stream()>>>(samples_dev_.data(), blocks_dev_.data());
    ), DALI_FAIL(make_string("Invalid input type: ", itype)););  // NOLINT(whitespace/parens)
  ), DALI_FAIL(make_string("Invalid output type: ", output_type_)););  // NOLINT(whitespace/parens)
}

DALI_REGISTER_OPERATOR(Cast, Cast<GPUBackend>, GPU);

}  // namespace dali
