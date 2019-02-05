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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_TRANSPOSE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_TRANSPOSE_H_

#include <vector>

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Transpose : public Operator<Backend> {
 public:
  explicit inline Transpose(const OpSpec &spec) :
    Operator<Backend>(spec),
    perm_(spec.GetRepeatedArgument<Index>("perm"))
    {}

  inline ~Transpose() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Transpose);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

 private:

  std::vector<Index> perm_;

  USE_OPERATOR_MEMBERS();

};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_TRANSPOSE_H_

