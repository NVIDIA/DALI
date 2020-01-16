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
  using SliceBase<Backend>::input_type_;
  using SliceBase<Backend>::output_type_;
  using SliceBase<Backend>::slice_anchors_;
  using SliceBase<Backend>::slice_shapes_;

  void RunImpl(Workspace<Backend> &ws) override {
    SliceBase<Backend>::RunImpl(ws);
  }

  void SetupSharedSampleParams(Workspace<Backend> &ws) override {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    SliceBase<Backend>::SetupSharedSampleParams(ws);
  }

  void DataDependentSetup(Workspace<Backend> &ws) override;

 private:
  inline TensorLayout GetDefaultLayout(int ndims) {
    switch (ndims) {
      case 2:
        return "HW";
      case 3:
        return "HWC";
      case 4:
        return "DHWC";
      default:
        return "";
    }
  }

  SliceAttr slice_attr_;

  static const int kImagesInId = 0;
  static const int kAnchorsInId = 1;
  static const int kSliceShapesInId = 2;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SLICE_SLICE_H_
