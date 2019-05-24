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

#ifndef DALI_PIPELINE_OPERATORS_CROP_NEW_SLICE_H_
#define DALI_PIPELINE_OPERATORS_CROP_NEW_SLICE_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/slice_base.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class NewSlice : public SliceBase<Backend> {
 public:
  explicit inline NewSlice(const OpSpec &spec)
    : SliceBase<Backend>(spec) {}

 protected:
  using SliceBase<Backend>::input_type_;
  using SliceBase<Backend>::output_type_;
  using SliceBase<Backend>::slice_anchors_;
  using SliceBase<Backend>::slice_shapes_;

  void RunImpl(Workspace<Backend> *ws, int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override {
    DALI_ENFORCE(ws->NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws->NumInput()));
    input_type_ = ws->template Input<Backend>(0).type().id();
    if (output_type_ == DALI_NO_TYPE) {
      output_type_ = input_type_;
    }
  }

  void DataDependentSetup(Workspace<Backend> *ws, int idx) override;

  void SetupSample(int data_idx,
                   const Dims &shape,
                   const float *anchor_norm,
                   const float *slice_dims_norm) {
    auto &anchor = slice_anchors_[data_idx];
    for (std::size_t d = 0; d < shape.size(); d++) {
      anchor[d] = anchor_norm[d] * shape[d];
    }

    // To decrease floating point error, first calculate the bounding box of crop and then
    // calculate the width and height having left and top coordinates
    auto &slice_shape = slice_shapes_[data_idx];
    for (std::size_t d = 0; d < shape.size(); d++) {
      float slice_end_norm = anchor_norm[d] + slice_dims_norm[d];
      int64_t slice_end = slice_end_norm * shape[d];
      slice_shape[d] = slice_end - anchor[d];
    }
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_NEW_SLICE_H_
