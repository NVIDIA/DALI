// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <string>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

std::string GetOpApi(const OpSpec &spec) {
  return spec.GetArgument<std::string>("_api");
}

std::string GetOpModule(const OpSpec &spec, ModuleSpecKind kind) {
  DALI_ENFORCE(kind != ModuleSpecKind::OpOnly, "This function can be only queried about module");
  std::string result;
  if (kind == ModuleSpecKind::LibApiModule) {
    result += "nvidia.dali.";
    result += GetOpApi(spec);
  } else if (kind == ModuleSpecKind::ApiModule) {
    result += GetOpApi(spec);
  }
  auto path = spec.GetSchema().ModulePath();
  if (result.size() && path.size()) {
    result += ".";
  }
  for (size_t i = 0; i < path.size(); i++) {
    result += path[i];
    if (i + 1 < path.size()) {
      result += ".";
    }
  }
  return result;
}

std::string GetOpDisplayName(const OpSpec &spec, ModuleSpecKind kind) {
  auto display_name = spec.GetArgument<std::string>("_display_name");
  if (kind == ModuleSpecKind::OpOnly) {
    return display_name;
  }
  auto module = GetOpModule(spec, kind);
  if (module.size()) {
    return module + "." + display_name;
  } else {
    return display_name;
  }
}

}  // namespace dali
