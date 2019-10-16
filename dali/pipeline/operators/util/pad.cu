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

#include <map>
#include <vector>
#include "dali/pipeline/operators/util/pad.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Pad)
    .DocStr(R"code(Pads all samples with `padding_value` in a given `axis`,
to match the size of the biggest dimension on this axis in the batch.
The element padding axis is specified with the argument `axis`.
Supported types: int, float.
Examples:
- Batch of 3 1-d samples, padding_value=-1, axis=0
  input  = [{3, 4, 2, 5, 4},
            {2, 2},
            {3, 199, 5}};
  output = [{3, 4, 2, 5, 4},
            {2, 2, -1, -1, -1},
            {3, 199, 5, -1, -1}]
- Batch of 2 2-d samples, padding_value=42, axis=1
  input  = [{{1, 2 , 3, 4},
             {5, 6, 7, 8}},
            {{1, 2},
             {4, 5}}]
  output = [{{1,  2,  3,  4},
             {5,  6,  7,  8}},
            {{1,  2, 42, 42},
             {4,  5, 42, 42}}]
)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("padding_value",
        R"code(The value to pad the batch with)code",
        DALI_FLOAT)
    .AddOptionalArg("axis",
        R"code(The axis on which the batch sample will be padded.
This arg is following zero-based indexing and 0 is the first axis
of the elements. If `axis` is -1, the padding will be done on all the axes.
)code", 0, false);

template <>
bool Pad<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                const DeviceWorkspace &ws) {
  output_desc.resize(1);
  const auto &input = ws.Input<GPUBackend>(0);
  std::size_t number_of_dims = input.shape().sample_dim();
  DALI_TYPE_SWITCH_WITH_FP16(input.type().id(), DataType,
    VALUE_SWITCH(number_of_dims, NumDims, (1, 2, 3, 4),
    (
      using Kernel = kernels::PadGPU<DataType, NumDims>;

      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();

      output_desc[0].type = TypeInfo::Create<DataType>();
      output_desc[0].shape.resize(batch_size_, NumDims);
      kmgr_.Initialize<Kernel>();

      auto in_view = view<const DataType, NumDims>(input);

      auto &req = kmgr_.Setup<Kernel>(0, ctx, in_view, axis_);
      output_desc[0].shape = req.output_shapes[0];
      // NOLINTNEXTLINE(whitespace/parens)
    ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims)););
  );  // NOLINT
  return true;
}


template <>
void Pad<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);
  std::size_t number_of_dims = input.shape().sample_dim();
  DALI_TYPE_SWITCH_WITH_FP16(input.type().id(), DataType,
    VALUE_SWITCH(number_of_dims, NumDims, (1, 2, 3, 4),
    (
      using Kernel = kernels::PadGPU<DataType, NumDims>;

      auto in_view = view<const DataType, NumDims>(input);
      auto out_view = view<DataType, NumDims>(output);
      kernels::KernelContext ctx;
      ctx.gpu.stream = ws.stream();
      kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, static_cast<DataType>(padding_value_));
      // NOLINTNEXTLINE(whitespace/parens)
    ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims)););
  );  // NOLINT
}

DALI_REGISTER_OPERATOR(Pad, Pad<GPUBackend>, GPU);

}  // namespace dali
