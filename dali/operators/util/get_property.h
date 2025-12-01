// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_GET_PROPERTY_H_
#define DALI_OPERATORS_UTIL_GET_PROPERTY_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class GetProperty : public StatelessOperator<Backend> {
 public:
  explicit GetProperty(const OpSpec &spec)
      : StatelessOperator<Backend>(spec),
        property_key_(spec.template GetArgument<std::string>("key")),
        property_reader_(GetPropertyReader(property_key_)) {}

 protected:
  bool HasContiguousOutputs() const override {
    return false;  // we may broadcast a common value to all samples
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return false;
  }

  void RunImpl(Workspace &ws) override {
    property_reader_(ws.Output<Backend>(0), ws);
  }

 private:
  using PropertyReaderFunc = void(TensorList<Backend> &, const Workspace &);
  using PropertyReader = std::function<PropertyReaderFunc>;

  std::string property_key_;
  PropertyReader property_reader_;

  static PropertyReader GetPropertyReader(std::string_view key);
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_GET_PROPERTY_H_
