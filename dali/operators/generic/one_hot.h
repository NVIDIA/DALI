// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_ONE_HOT_H
#define DALI_OPERATORS_RANDOM_ONE_HOT_H

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

class OneHot : public Operator<CPUBackend> {
 public:
  inline explicit OneHot(const OpSpec &spec)
      : Operator<CPUBackend>(spec), nclasses_(spec.GetArgument<int64_t>("nclasses")) {}

  inline ~OneHot() override = default;

  DISABLE_COPY_MOVE_ASSIGN(OneHot);

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    output_desc.resize(1);
    output_desc[0].shape = uniform_list_shape(batch_size_, {nclasses_});
    TYPE_SWITCH(output_type_, type2id, DType, PREEMPH_TYPES, (
            {
              TypeInfo type;
              type.SetType<DType>(output_type_);
              output_desc[0].type = type;
            }
    ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
    return true;
  }

  void RunImpl(HostWorkspace &ws) override;

  const DALIDataType output_type_;
  int nclasses_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_ONE_HOT_H