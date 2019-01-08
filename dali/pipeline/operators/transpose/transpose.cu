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

#include "dali/pipeline/operators/transpose/transpose.h"
#include "dali/error_handling.h"

namespace dali {


namespace {
  void NaiveTransposeKernel(const TensorList<GPUBackend>& input, TensorList<GPUBackend>* output) {

  }
  void cuTTKernel(const TensorList<GPUBackend>& input, TensorList<GPUBackend>* output) {

  }
}  // namespace


template<>
void Transpose<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto *output = ws->Output<GPUBackend>(idx);

  if (input.IsDenseTensor()) {
    cuTTKernel(input, output);
  } else {
    NaiveTransposeKernel(input, output);
  }

  // TODO(spanev): implem
}

DALI_REGISTER_OPERATOR(Transpose, Transpose<GPUBackend>, GPU);

}  // namespace dali

