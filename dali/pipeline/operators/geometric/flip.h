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

#ifndef DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_H_
#define DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_H_

#include <vector>
#include <string>
#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Flip: public Operator<Backend> {
 public:
  explicit Flip(const OpSpec &spec);

  ~Flip() override = default;
  DISABLE_COPY_MOVE_ASSIGN(Flip);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  int GetHorizontal(const ArgumentWorkspace *ws, int idx) {
    return this->spec_.template GetArgument<int>("horizontal", ws, idx);
  }

  int GetVertical(const ArgumentWorkspace *ws, int idx) {
    return this->spec_.template GetArgument<int>("vertical", ws, idx);
  }

  std::vector<int> GetHorizontal(const ArgumentWorkspace *ws) {
    return GetTensorArgument(ws, "horizontal");
  }

  std::vector<int> GetVertical(const ArgumentWorkspace *ws) {
    return GetTensorArgument(ws, "vertical");
  }

 private:
  std::vector<int> GetTensorArgument(const ArgumentWorkspace *ws, const std::string &name) {
    std::vector<int> result(this->batch_size_);
    if (this->spec_.HasTensorArgument(name)) {
      auto &arg = ws->ArgumentInput(name);
      auto *ptr = arg.data<int>();
      DALI_ENFORCE(arg.size() == this->batch_size_);
      std::copy(ptr, ptr + arg.size(), result.begin());
    } else {
      auto value = this->spec_.template GetArgument<int>(name, ws, 0);
      std::fill(std::begin(result), std::end(result), value);
    }
    return result;
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_GEOMETRIC_FLIP_H_
