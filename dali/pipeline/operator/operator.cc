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

#include "dali/pipeline/operator/operator.h"

namespace dali {

DALI_DEFINE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DEFINE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);

std::unique_ptr<OperatorBase> InstantiateOperator(const OpSpec &spec) {
  string device = spec.GetArgument<string>("device");
  // traverse devices by likelihood (gpu, cpu, mixed, support)
  if (device == "gpu") {
    return GPUOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else if (device == "cpu") {
    return CPUOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else if (device == "mixed") {
    return MixedOperatorRegistry::Registry().Create(spec.name(), spec, &device);
  } else {
    DALI_FAIL("Unknown device: " + device);
  }
}

}  // namespace dali
