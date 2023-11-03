// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/operators/util/property.h"
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
        property_(PropertyFactory()) {}

  ~GetProperty() override = default;
  DISABLE_COPY_MOVE_ASSIGN(GetProperty);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    output_desc.resize(1);
    output_desc[0].shape = property_->GetShape(input);
    output_desc[0].type = property_->GetType(input);
    return true;
  }

  void RunImpl(Workspace &ws) override {
    property_->FillOutput(ws);
  }

 private:
  std::unique_ptr<tensor_property::Property<Backend>> PropertyFactory() {
    if (property_key_ == "source_info") {
      return std::make_unique<tensor_property::SourceInfo<Backend>>();
    } else if (property_key_ == "layout") {
      return std::make_unique<tensor_property::Layout<Backend>>();
    } else {
      DALI_FAIL(make_string("Unknown property key: ", property_key_));
    }
  }

  const std::string property_key_;
  std::unique_ptr<tensor_property::Property<Backend>> property_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_GET_PROPERTY_H_
