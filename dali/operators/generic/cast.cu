// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/core/span.h"
#include "dali/kernels/common/cast.cuh"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/operators/generic/cast.h"


namespace dali {

class CastGPU : public Cast<GPUBackend> {
 public:
  explicit CastGPU(const OpSpec &spec) : Cast<GPUBackend>{spec} {}
  void RunImpl(Workspace &ws) override;
  ~CastGPU() override = default;

 private:
  static constexpr int kBlockSize = 1024;
  static constexpr int kLogicalBlockSize = 1024;
  USE_OPERATOR_MEMBERS();
};

void CastGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  int num_samples = input.num_samples();
  const auto& in_sh = input.shape();
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  kernels::DynamicScratchpad scratchpad({}, ws.stream());
  using SampleDesc = kernels::cast::SampleDesc;
  SampleDesc *samples = scratchpad.AllocatePinned<SampleDesc>(num_samples);
  uint32_t offset_blk = 0;
  for (int i = 0; i < num_samples; i++) {
    auto &sample = samples[i];
    sample.output = output.raw_mutable_tensor(i);
    sample.input = input.raw_tensor(i);
    sample.first_block = offset_blk;
    sample.sample_size = in_sh.tensor_size(i);
    offset_blk += div_ceil(sample.sample_size, kLogicalBlockSize);
  }

  auto *samples_dev =
      scratchpad.ToGPU(ws.stream(), span<const SampleDesc>(samples, num_samples));

  TYPE_SWITCH(output.type(), type2id, Out, CAST_ALLOWED_TYPES, (
    TYPE_SWITCH(input.type(), type2id, In, CAST_ALLOWED_TYPES, (
      kernels::cast::BinSearchCastKernel<Out, In>
          <<<offset_blk, kBlockSize, 0, ws.stream()>>>(samples_dev, num_samples, kLogicalBlockSize);
    ), DALI_FAIL(make_string("Invalid input type: ", input.type())););  // NOLINT(whitespace/parens)
  ), DALI_FAIL(make_string("Invalid output type: ", output.type())););  // NOLINT(whitespace/parens)
  CUDA_CALL(cudaGetLastError());
}

DALI_REGISTER_OPERATOR(Cast, CastGPU, GPU);
DALI_REGISTER_OPERATOR(CastLike, CastGPU, GPU);

}  // namespace dali
