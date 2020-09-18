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

#ifndef DALI_OPERATORS_GENERIC_SLICE_SLICE_H_
#define DALI_OPERATORS_GENERIC_SLICE_SLICE_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/generic/slice/slice_base.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class Slice : public SliceBase<Backend> {
 public:
  explicit inline Slice(const OpSpec &spec)
    : SliceBase<Backend>(spec)
    , slice_attr_(spec) {}

 protected:
  void ProcessCroppingAttrs(const workspace_t<Backend> &ws) override {
    slice_attr_.ProcessArguments<Backend>(ws);
  }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const override {
    return slice_attr_.GetCropWindowGenerator(data_idx);
  }

 private:
  SliceAttr slice_attr_;

  static const int kImagesInId = 0;
  static const int kAnchorsInId = 1;
  static const int kSliceShapesInId = 2;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_H_
