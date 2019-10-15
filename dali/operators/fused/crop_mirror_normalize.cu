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

#include <utility>
#include <vector>
#include "dali/operators/fused/crop_mirror_normalize.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_gpu.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

template <>
bool CropMirrorNormalize<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                const DeviceWorkspace &ws) {
  output_desc.resize(1);
  SetupAndInitialize(ws);
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);
  std::size_t number_of_dims = input.shape().sample_dim();
  DALI_TYPE_SWITCH_WITH_FP16(input_type_, InputType,
    DALI_TYPE_SWITCH_WITH_FP16(output_type_, OutputType,
      VALUE_SWITCH(number_of_dims, Dims, (3, 4, 5),
      (
        using Kernel = kernels::SliceFlipNormalizePermutePadGPU<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
        output_desc[0].type = TypeInfo::Create<OutputType>();
        output_desc[0].shape.resize(batch_size_, Dims);
        kmgr_.Initialize<Kernel>();

        kernels::KernelContext ctx;
        ctx.gpu.stream = ws.stream();
        auto in_view = view<const InputType, Dims>(input);
        auto &req = kmgr_.Setup<Kernel>(0, ctx, in_view, kernel_sample_args);
        output_desc[0].shape = req.output_shapes[0];
        // NOLINTNEXTLINE(whitespace/parens)
      ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims)););
    )
  );  // NOLINT(whitespace/parens)
  return true;
}

template<>
void CropMirrorNormalize<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto &output = ws.Output<GPUBackend>(0);

  std::size_t number_of_dims = input.shape().sample_dim();
  DALI_TYPE_SWITCH_WITH_FP16(input_type_, InputType,
    DALI_TYPE_SWITCH_WITH_FP16(output_type_, OutputType,
      VALUE_SWITCH(number_of_dims, Dims, (3, 4, 5),
      (
        using Kernel = kernels::SliceFlipNormalizePermutePadGPU<OutputType, InputType, Dims>;
        using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
        auto in_view = view<const InputType, Dims>(input);
        auto out_view = view<OutputType, Dims>(output);
        kernels::KernelContext ctx;
        ctx.gpu.stream = ws.stream();
        auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
        kmgr_.Run<Kernel>(0, 0, ctx, out_view, in_view, kernel_sample_args);
        // NOLINTNEXTLINE(whitespace/parens)
      ), DALI_FAIL("Not supported number of dimensions: " + std::to_string(number_of_dims)););
    )
  )
}

DALI_REGISTER_OPERATOR(CropMirrorNormalize, CropMirrorNormalize<GPUBackend>, GPU);

}  // namespace dali
