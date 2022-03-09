// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_H_

#include <tuple>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/scratch.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/generic/slice/slice_base.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Crop : public SliceBase<Backend> {
 public:
  explicit inline Crop(const OpSpec &spec) :
    SliceBase<Backend>(spec),
    crop_attr_(spec) {}

 protected:
  void ProcessCroppingAttrs(const OpSpec &spec, const workspace_t<Backend> &ws) override {
    crop_attr_.ProcessArguments(spec, ws);
  }

  const CropWindowGenerator &GetCropWindowGenerator(std::size_t data_idx) const override {
    return crop_attr_.GetCropWindowGenerator(data_idx);
  }

  CropAttr crop_attr_;
  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_H_
