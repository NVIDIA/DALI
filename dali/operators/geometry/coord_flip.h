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

#ifndef DALI_OPERATORS_GEOMETRY_COORD_FLIP_H_
#define DALI_OPERATORS_GEOMETRY_COORD_FLIP_H_

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
    for (int i = 1; i < in_shape.size(); i++) {
      DALI_ENFORCE(ndim_ == in_shape[i][1],
          make_string(
              "All samples are expected to have coordinates with same number of dimensions. Got : ",
              in_shape));
    }
    DALI_ENFORCE(ndim_ <= 3, make_string("Unexpected number of dimensions ", ndim_));
    if (layout_.empty()) {
      switch (ndim_) {
        case 1:
          layout_ = "x";
          break;
        case 2:
          layout_ = "xy";
          break;
        case 3:
          layout_ = "xyz";
          break;
        default:
          layout_ = "";
          break;
      }
    }

    x_dim_ = 0;
    if (ndim_ > 0) {
      x_dim_ = layout_.find('x');
      DALI_ENFORCE(x_dim_ >= 0, "Dimension \"x\" not found in the layout");
    }

    y_dim_ = 1;
    if (ndim_ > 1) {
      y_dim_ = layout_.find('y');
      DALI_ENFORCE(y_dim_ >= 0, "Dimension \"y\" not found in the layout");
    }

    z_dim_ = 2;
    if (ndim_ > 2) {
      z_dim_ = layout_.find('z');
      DALI_ENFORCE(z_dim_ >= 0, "Dimension \"z\" not found in the layout");
    }

    return true;
  }

  // Layout of the coordinates
  TensorLayout layout_;
  // Number of dimensions
  int ndim_ = -1;
  // Indices of x, y and z dimensions
  int x_dim_ = -1, y_dim_ = -1, z_dim_ = -1;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GEOMETRY_COORD_FLIP_H_
