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

template <typename Backend>
class BbFlip;

template <>
class BbFlip<CPUBackend> : public Operator<CPUBackend> {
 public:
  explicit BbFlip(const OpSpec &spec);

  ~BbFlip() override = default;
  DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override;
  using Operator<CPUBackend>::RunImpl;

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
