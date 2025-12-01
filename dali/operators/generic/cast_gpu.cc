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

#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/cast_gpu.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/operators/generic/cast.h"


namespace dali {

class CastGPU : public Cast<GPUBackend> {
 public:
  explicit CastGPU(const OpSpec &spec) : Cast<GPUBackend>{spec} {}
  void RunImpl(Workspace &ws) override;
  ~CastGPU() override = default;

  USE_OPERATOR_MEMBERS();
};

void CastGPU::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();
  kernels::DynamicScratchpad scratchpad(ws.stream());
  ctx.scratchpad = &scratchpad;

  TYPE_SWITCH(output.type(), type2id, Out, CAST_ALLOWED_TYPES, (
    TYPE_SWITCH(input.type(), type2id, In, CAST_ALLOWED_TYPES, (
      auto kernel = kernels::cast::CastGPU<Out, In>{};
      kernel.Run(ctx, flatten(view<Out>(output)), flatten(view<const In>(input)));
    ), DALI_FAIL(make_string("Invalid input type: ", input.type())););  // NOLINT(whitespace/parens)
  ), DALI_FAIL(make_string("Invalid output type: ", output.type())););  // NOLINT(whitespace/parens)
}

DALI_REGISTER_OPERATOR(Cast, CastGPU, GPU);
DALI_REGISTER_OPERATOR(CastLike, CastGPU, GPU);

}  // namespace dali
