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

#ifndef DALI_PIPELINE_OPERATORS_UTIL_CONSTANT_H_
#define DALI_PIPELINE_OPERATORS_UTIL_CONSTANT_H_

#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Constant : public Operator<Backend> {
 public:
  explicit inline Constant(const OpSpec &spec) :
    Operator<Backend>(spec) {
    auto output_type = spec.GetArgument<DALIDataType>("source_dtype");
    auto output_shape = spec.GetRepeatedArgument<Index>("source_shape");
    void *output_data = spec.GetArgument<void *>("source_data");
    Tensor<CPUBackend> t;
    auto type = TypeTable::GetTypeInfo(output_type);
    const size_t bytes = type.size() * Product(output_shape);
    t.ShareData(output_data, bytes, type, output_shape);
    source_.Copy(t, 0);
    first_iter_.resize(batch_size_);
    for (size_t i = 0; i < first_iter_.size(); ++i) {
      first_iter_[i] = true;
    }
  }

  virtual inline ~Constant() = default;

  DISABLE_COPY_MOVE_ASSIGN(Constant);

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

 private:
  Tensor<CPUBackend> source_;
  vector<bool> first_iter_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_CONSTANT_H_
