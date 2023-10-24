// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_CROP_MIRROR_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_CROP_MIRROR_H_

#include <random>
#include <vector>
#include <utility>
#include <cmath>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/image/transform.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {

class ResizeCropMirrorAttr : public ResizeAttr, protected CropAttr {
 public:
  explicit inline ResizeCropMirrorAttr(const OpSpec &spec)
    : CropAttr(spec)
    , mirror_("mirror", spec) {}

 protected:
  void SetFlagsAndMode(const OpSpec &spec) override {
    ResizeAttr::SetFlagsAndMode(spec);
    has_roi_ = !IsWholeImage();
  }

  void GetRoI(const OpSpec &spec,
              const ArgumentWorkspace &ws,
              const TensorListShape<> input_shape) override {
    mirror_.Acquire(spec, ws, batch_size_);
    if (has_roi_) {
      roi_relative_ = false;
      roi_start_.resize(batch_size_ * spatial_ndim_);
      roi_end_.resize(batch_size_ * spatial_ndim_);
      for (int i = 0; i < batch_size_; i++) {
        CropAttr::ProcessArguments(spec, &ws, i);
        int mirror = *mirror_[i].data;
        auto wnd = GetCropWindowGenerator(i)(input_shape[i], layout_);

        for (int d = 0; d < spatial_ndim_; d++) {
          auto &lo = roi_start_[i * spatial_ndim_ + d];
          auto &hi = roi_end_[i * spatial_ndim_ + d];
          lo = wnd.anchor[d];
          hi = lo + wnd.shape[d];
          if (mirror & (i << d))
            std::swap(lo, hi);
        }
      }
    }
  }

  ArgValue<int, 0> mirror_;
};

template <typename Backend>
class ResizeCropMirror : public StatelessOperator<Backend>
                       , protected ResizeBase<Backend> {
 public:
  explicit ResizeCropMirror(const OpSpec &spec);

 protected:
  int NumSpatialDims() const { return resize_attr_.spatial_ndim_; }
  int FirstSpatialDim() const { return resize_attr_.first_spatial_dim_; }

  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  void RunImpl(Workspace &ws) override;

  void PrepareParams(const ArgumentWorkspace &ws, const TensorListShape<> &input_shape,
                     const TensorLayout &layout) {
    resize_attr_.PrepareResizeParams(spec_, ws, input_shape, layout);
    assert(NumSpatialDims() >= 1 && NumSpatialDims() <= 3);
    assert(FirstSpatialDim() >= 0);
    int N = input_shape.num_samples();
    resample_params_.resize(N * NumSpatialDims());
    resampling_attr_.PrepareFilterParams(spec_, ws, N);
    resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                         make_cspan(resize_attr_.params_));
  }

  void InitializeBackend();

  USE_OPERATOR_MEMBERS();
  std::vector<kernels::ResamplingParams> resample_params_;
  TensorList<CPUBackend> attr_staging_;
  using Operator<Backend>::RunImpl;

  ResizeCropMirrorAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_CROP_MIRROR_H_
