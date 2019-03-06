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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_

#include <string>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class DumpImage : public Operator<Backend> {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator<Backend>(spec),
    suffix_(spec.GetArgument<string>("suffix")) {
    DALI_ENFORCE(spec.GetArgument<DALITensorLayout>("input_layout") == DALI_NHWC,
        "CHW not supported yet.");
  }

  inline ~DumpImage() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  const string suffix_;
};
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_DUMP_IMAGE_H_
