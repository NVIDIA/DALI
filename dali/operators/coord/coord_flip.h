// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_COORD_COORD_FLIP_H_
#define DALI_OPERATORS_COORD_COORD_FLIP_H_

#include <string>
#include <vector>

#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class CoordFlip : public Operator<Backend> {
 public:
  explicit CoordFlip(const OpSpec &spec)
      : Operator<Backend>(spec)
      , layout_(spec.GetArgument<TensorLayout>("layout")) {}

  ~CoordFlip() override = default;
  DISABLE_COPY_MOVE_ASSIGN(CoordFlip);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    DALI_ENFORCE(input.type().id() == DALI_FLOAT, "Input is expected to be float");

    output_desc.resize(1);
    auto in_shape = input.shape();
    output_desc[0].shape = in_shape;
    output_desc[0].type = input.type();

    DALI_ENFORCE(in_shape[0].size() == 2);
    ndim_ = in_shape[0][1];
    DALI_ENFORCE(ndim_ >= 1 && ndim_ <= 3, make_string("Unexpected number of dimensions ", ndim_));

    if (layout_.empty()) {
      switch (ndim_) {
        case 1:
          layout_ = "x";
          break;
        case 2:
          layout_ = "xy";
          break;
        case 3:
        default:
          layout_ = "xyz";
          break;
      }
    }
    return true;
  }

  TensorLayout layout_;
  int ndim_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_COORD_COORD_FLIP_H_
