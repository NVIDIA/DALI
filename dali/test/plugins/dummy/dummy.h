// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_PLUGINS_DUMMY_DUMMY_H_
#define DALI_TEST_PLUGINS_DUMMY_DUMMY_H_

#include <cstring>
#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace other_ns {

template <typename Backend>
class Dummy : public ::dali::Operator<Backend> {
 public:
  inline explicit Dummy(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {}

  virtual inline ~Dummy() = default;

  DISABLE_COPY_MOVE_ASSIGN(Dummy);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    output_desc.resize(1);
    output_desc[0] = {input.shape(), input.type()};
    return true;
  }

  void RunImpl(::dali::workspace_t<Backend> &ws) override;
};

}  // namespace other_ns

#endif  // DALI_TEST_PLUGINS_DUMMY_DUMMY_H_
