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

#ifndef DALI_OPERATORS_BBOX_BBOX_PASTE_H_
#define DALI_OPERATORS_BBOX_BBOX_PASTE_H_

#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class BBoxPaste : public Operator<Backend> {
 public:
  explicit inline BBoxPaste(const OpSpec &spec) :
    Operator<Backend>(spec) {
    use_ltrb_ = spec.GetArgument<bool>("ltrb");
  }

 protected:
  bool use_ltrb_ = false;

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_BBOX_BBOX_PASTE_H_
