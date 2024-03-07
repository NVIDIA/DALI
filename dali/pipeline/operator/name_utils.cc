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

std::string GetOpModule(const OpSpec &spec) {
  return spec.GetArgument<std::string>("_module");;
}

std::string GetOpDisplayName(const OpSpec &spec, bool include_module_path) {
  auto display_name = spec.GetArgument<std::string>("_display_name");
  if (!include_module_path) {
    return display_name;
  }
  auto module = GetOpModule(spec);
  if (module.size()) {
    return module + "." + display_name;
  } else {
    return display_name;
  }
}

}  // namespace dali
