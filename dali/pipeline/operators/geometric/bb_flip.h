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

#ifndef DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_
#define DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_

#include <dali/pipeline/operators/common.h>
#include <dali/pipeline/operators/operator.h>
#include <string>

namespace dali {
//
//class BoundingBox {
// public:
//  explicit BoundingBox(float x1, float y1, float w1, float h1, bool ltrb = false)
//      : x{x1}, y{y1}, w{ltrb ? w1 - x1 : w1}, h{ltrb ? h1 - y1 : h1} {
//    DALI_ENFORCE(x >= 0, "Expected x >= 0. Received: " + std::to_string(x));
//    DALI_ENFORCE(y >= 0, "Expected y >= 0. Received: " + std::to_string(y));
//    DALI_ENFORCE(w >= 0, "Expected w >= 0. Received: " + std::to_string(w));
//    DALI_ENFORCE(h >= 0, "Expected h >= 0. Received: " + std::to_string(h));
//    DALI_ENFORCE(x + w <= 1, "Expected x + w <= 1. Received: " + std::to_string(x + w));
//    DALI_ENFORCE(y + h <= 1, "Expected y + h <= 1. Received: " + std::to_string(y + h));
//  }
//
//  BoundingBox(const BoundingBox &other)
//      : x{other.x}, y{other.y}, w{other.w}, h{other.h} {}
//
//  BoundingBox HFlip(bool on) const {
//    return on ? BoundingBox(1 - (x + w), y, w, h) : *this;
//  }
//
//  BoundingBox VFlip(bool on) const {
//    return on ? BoundingBox(x, 1 - (y + h), w, h) : *this;
//  }
//
//  std::array<float, 4> Coordinates(bool ltrb) const {
//    return ltrb ? std::array<float, 4>{x, y, x + w, y + h}
//                : std::array<float, 4>{x, y, w, h};
//  }
//
// private:
//  float x, y, w, h;
//};
class BbFlip : public Operator<CPUBackend> {
 public:
  explicit BbFlip(const OpSpec &spec);

  virtual ~BbFlip() = default;
  DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override;

 private:
  /**
   * Checks, if argument provided by user is a scalar and,
   * in such case, extends this scalar to entire tensor
   * @tparam TensorDataType Underlying data type in tensor
   */
  template <typename TensorDataType>
  void ExtendScalarToTensor(std::string argument_name, const OpSpec &spec,
                            Tensor<CPUBackend> *tensor) {
    tensor->Resize({batch_size_});
    for (int i = 0; i < batch_size_; i++) {
      tensor->mutable_data<TensorDataType>()[i] =
          spec.GetArgument<TensorDataType>(argument_name);
    }
  }

  /**
   * Bounding box can be represented in two ways:
   * 1. Upper-left corner, width, height (`wh_type`)
   *    (x1, y1,  w,  h)
   * 2. Upper-left and Lower-right corners (`two-point type`)
   *    (x1, y1, x2, y2)
   *
   * Both of them have coordinates in image coordinate system
   * (i.e. 0.0-1.0)
   *
   * If `coordinates_type_ltrb_` is true, then we deal with 2nd type. Otherwise,
   * the 1st one.
   */
  const bool ltrb_;

  /**
   * If true, flip is performed along vertical (x) axis
   */
  Tensor<CPUBackend> flip_type_vertical_;

  /**
   * If true, flip is performed along horizontal (y) axis
   */
  Tensor<CPUBackend> flip_type_horizontal_;

  /**
   * XXX: This is workaround for architectural mishap, that there are 2 access
   * points for operator arguments: Workspace and OpSpec
   */
  bool vflip_is_tensor_, hflip_is_tensor_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_GEOMETRIC_BB_FLIP_H_
