// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <>
void Operator<SupportBackend>::ReleaseInputs(SupportWorkspace *ws) {
  // TODO(slayton): Support operators are likely to have input support in
  // the future - this method will have to be updated to reflect that
  return;
}

template <>
void Operator<CPUBackend>::ReleaseInputs(SampleWorkspace *ws) {
  for (int i = 0; i < ws->NumInput(); ++i) {
    if (ws->InputIsType<CPUBackend>(i)) {
      ws->Input<CPUBackend>(i).release();
    } else {
      ws->Input<GPUBackend>(i).release();
    }
  }
}

template <>
void Operator<GPUBackend>::ReleaseInputs(DeviceWorkspace *ws) {
  for (int i = 0; i < ws->NumInput(); ++i) {
    if (ws->InputIsType<CPUBackend>(i)) {
      ws->Input<CPUBackend>(i).release(ws->stream());
    } else {
      ws->Input<GPUBackend>(i).release(ws->stream());
    }
  }
}

DALI_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(SupportOperator, OperatorBase);

}  // namespace dali
