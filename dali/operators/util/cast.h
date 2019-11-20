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

#ifndef DALI_OPERATORS_UTIL_CAST_H_
#define DALI_OPERATORS_UTIL_CAST_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"
#include "dali/core/convert.h"

namespace dali {


#define CAST_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, \
  float16, float, double)

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
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    return false;
  }

  void RunImpl(Workspace<Backend> &ws) override;

 private:
  template <typename OType, typename IType>
  inline void CPUHelper(OType *out, const IType *in, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      out[i] = ConvertSat<OType>(in[i]);
    }
  }

  DALIDataType output_type_;

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_CAST_H_
