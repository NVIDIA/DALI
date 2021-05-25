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

#include "dali/operators/audio/mel_scale/mel_filter_bank.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/audio/mel_scale/mel_filter_bank_gpu.h"

namespace dali {

template <>
bool MelFilterBank<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                          const workspace_t<GPUBackend> &ws) {
  output_desc.resize(kNumOutputs);
  const auto &input = ws.InputRef<GPUBackend>(0);
  const auto &in_shape = input.shape();
  auto layout = input.GetLayout();
  auto ndim = in_shape.sample_dim();
  args_.axis = layout.empty() ? std::max(0, ndim - 2) : layout.find('f');
  DALI_ENFORCE(args_.axis >= 0 && args_.axis < ndim,
    make_string("'f' axis not present in the layout. Got: `", layout, "`"));
  ctx_.gpu.stream = ws.stream();
  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    using MelFilterBankKernel = kernels::audio::MelFilterBankGpu<T>;
    kmgr_.Initialize<MelFilterBankKernel>();
    kmgr_.Resize<MelFilterBankKernel>(1, 1);
    output_desc[0].type = TypeTable::GetTypeInfo(TypeTable::GetTypeID<T>());
    auto in_view = view<const T>(input);
    auto &req = kmgr_.Setup<MelFilterBankKernel>(0, ctx_, in_view, args_);
    output_desc[0].shape = req.output_shapes[0];
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
  return true;
}

template <>
void MelFilterBank<GPUBackend>::RunImpl(workspace_t<GPUBackend> &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto &output = ws.OutputRef<GPUBackend>(0);
  const auto &in_shape = input.shape();
  ctx_.gpu.stream = ws.stream();
  TYPE_SWITCH(input.type().id(), type2id, T, MEL_FBANK_SUPPORTED_TYPES, (
    using MelFilterBankKernel = kernels::audio::MelFilterBankGpu<T>;
    auto in_view = view<const T>(input);
    auto out_view = view<T>(output);
    kmgr_.Run<MelFilterBankKernel>(0, 0, ctx_, out_view, in_view);
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT
}

DALI_REGISTER_OPERATOR(MelFilterBank, MelFilterBank<GPUBackend>, GPU);

}  // namespace dali
