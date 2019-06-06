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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_CAST_H_
#define DALI_PIPELINE_OPERATORS_UTIL_CAST_H_

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Cast : public Operator<Backend> {
 public:
  explicit inline Cast(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("dtype"))
    {}

  inline ~Cast() override = default;

  DISABLE_COPY_MOVE_ASSIGN(Cast);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

 private:
  template <typename IType, typename OType>
  inline void CPUHelper(OType * out, const IType * in, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      out[i] = static_cast<OType>(in[i]);
    }
  }

  DALIDataType output_type_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_CAST_H_
