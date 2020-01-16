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

#ifndef DALI_OPERATORS_DEBUG_DUMP_IMAGE_H_
#define DALI_OPERATORS_DEBUG_DUMP_IMAGE_H_

#include <string>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class DumpImage : public Operator<Backend> {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator<Backend>(spec),
    suffix_(spec.GetArgument<string>("suffix")) {
    DALI_ENFORCE(spec.GetArgument<TensorLayout>("input_layout") == "HWC",
        "CHW not supported yet.");
  }

  inline ~DumpImage() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

  const string suffix_;
};
}  // namespace dali

#endif  // DALI_OPERATORS_DEBUG_DUMP_IMAGE_H_
