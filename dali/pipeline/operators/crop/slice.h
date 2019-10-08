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

#ifndef DALI_PIPELINE_OPERATORS_CROP_SLICE_H_
#define DALI_PIPELINE_OPERATORS_CROP_SLICE_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/crop/slice_base.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class Slice : public SliceBase<Backend> {
 public:
  explicit inline Slice(const OpSpec &spec)
    : SliceBase<Backend>(spec) {}

 protected:
  using SliceBase<Backend>::input_type_;
  using SliceBase<Backend>::output_type_;
  using SliceBase<Backend>::slice_anchors_;
  using SliceBase<Backend>::slice_shapes_;

  void RunImpl(Workspace<Backend> &ws) override;

  void SetupSharedSampleParams(Workspace<Backend> &ws) override {
    DALI_ENFORCE(ws.NumInput() == 3,
      "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
    SliceBase<Backend>::SetupSharedSampleParams(ws);
  }

  void DataDependentSetup(Workspace<Backend> &ws) override;

  void SetupSample(int data_idx,
                   TensorLayout layout,
                   const kernels::TensorShape<> &img_shape,
                   const int64_t args_ndims,
                   const float *anchor_norm,
                   const float *slice_dims_norm) {
    auto &anchor = slice_anchors_[data_idx];
    anchor = std::vector<int64_t>(img_shape.size(), 0);
    auto &slice_shape = slice_shapes_[data_idx];
    slice_shape = std::vector<int64_t>(img_shape.begin(), img_shape.end());

    // If only two dimensions are provided (old API style)
    // we calculate the position of the dimensions based on
    // layout.
    if (args_ndims == 2 && img_shape.size() > 2) {
      int i_h = layout.find('H');
      int i_w = layout.find('W');
      DALI_ENFORCE(i_h >= 0, "The layout \"" + layout.str() +
                   "\" does not define height dimension (H)");
      DALI_ENFORCE(i_w >= 0, "The layout \"" + layout.str() +
                   "\" does not define width dimension (W)");
      for (int d = 0; d < img_shape.size(); d++) {
        anchor[d] = 0;
        slice_shape[d] = img_shape[d];
      }

      // TODO(janton): In Slice API we receive coordinates in XY format
      anchor[i_w] = anchor_norm[0] * img_shape[i_w];
      anchor[i_h] = anchor_norm[1] * img_shape[i_h];

      float slice_end_norm_w = anchor_norm[0] + slice_dims_norm[0];
      slice_shape[i_w] = slice_end_norm_w * img_shape[i_w] - anchor[i_w];

      float slice_end_norm_h = anchor_norm[1] + slice_dims_norm[1];
      slice_shape[i_h] = slice_end_norm_h * img_shape[i_h] - anchor[i_h];

    } else {
      // General case expects same number of dimensions in the
      // slice arguments as in the input image

      // To decrease floating point error, first calculate the end of the
      // bounding box and then calculate the shape
      for (int d = 0; d < img_shape.size(); d++) {
        anchor[d] = anchor_norm[d] * img_shape[d];
        float slice_end_norm = anchor_norm[d] + slice_dims_norm[d];
        int64_t slice_end = slice_end_norm * img_shape[d];
        slice_shape[d] = slice_end - anchor[d];
      }
    }
  }

 private:
  static const int kImagesInId = 0;
  static const int kAnchorsInId = 1;
  static const int kSliceShapesInId = 2;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_SLICE_H_
