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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_
#define DALI_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_

#include <random>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/image/transform.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief Performs fused resize+crop+mirror
 */

template <typename Backend>
class ResizeCropMirror : public Operator<CPUBackend> {
 public:
  explicit inline ResizeCropMirror(const OpSpec &spec) :
    Operator(spec) {
    vector<int> temp_crop;
    try {
      temp_crop = spec.GetRepeatedArgument<int>("crop");
      if (temp_crop.size() == 1) {
        temp_crop.push_back(temp_crop.back());
      }
    } catch (std::runtime_error e) {
      try {
        int temp = spec.GetArgument<int>("crop");
        temp_crop = {temp, temp};
      } catch (std::runtime_error e) {
        DALI_FAIL("Invalid type of argument \"crop\". Expected int or list of int");
      }
    }

    DALI_ENFORCE(temp_crop.size() == 2, "Argument \"crop\" expects a list of at most 2 elements, "
        + to_string(temp_crop.size()) + " given.");
    crop_h_ = temp_crop[0];
    crop_w_ = temp_crop[1];
    // Validate input parameters
    resize_shorter_ = (spec.HasArgument("resize_shorter") ||
                       spec.HasTensorArgument("resize_shorter"));
    resize_x_ = spec.HasArgument("resize_x") ||
                spec.HasTensorArgument("resize_x");
    resize_y_ = spec.HasArgument("resize_y") ||
                spec.HasTensorArgument("resize_y");
    DALI_ENFORCE(resize_shorter_ != (resize_x_ || resize_y_),
                 "Options `resize_shorter` and `resize_x` or `resize_y` are mutually exclusive.");
    DALI_ENFORCE(crop_h_ > 0 && crop_w_ > 0);

    // Resize per-image & per-thread data
    tl_workspace_.resize(num_threads_);

    // per-image-set data
    per_thread_meta_.resize(num_threads_);
  }

  virtual inline ~ResizeCropMirror() = default;

 protected:
  using TransformMeta = struct {
    int H, W, C;
    int rsz_h, rsz_w;
    int crop_x, crop_y;
    int mirror;
  };

  inline void SetupSharedSampleParams(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);

    // enforce that all shapes match
    for (int i = 1; i < ws->NumInput(); ++i) {
      DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }

    per_thread_meta_[ws->thread_idx()] = GetTransformMeta(input.shape(), ws, ws->data_idx());
  }

  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);
    DALI_ENFORCE(input.ndim() == 3);
    DALI_ENFORCE(IsType<uint8>(input.type()),
        "Expects input data in uint8.");
    DALI_ENFORCE(input.dim(2) == 1 || input.dim(2) == 3,
        "ResizeCropMirror supports hwc rgb & grayscale inputs.");

    const TransformMeta &meta = per_thread_meta_[ws->thread_idx()];

    // Resize the output & run
    output->Resize({crop_h_, crop_w_, meta.C});

    tl_workspace_[ws->thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    DALI_CALL(ResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        DALI_INTERP_LINEAR,
        tl_workspace_[ws->thread_idx()].data()));
  }

  inline const TransformMeta GetTransformMeta(const vector<Index> &input_shape,
                                              SampleWorkspace * ws, const Index index) {
    TransformMeta meta;
    meta.H = input_shape[0];
    meta.W = input_shape[1];
    meta.C = input_shape[2];

    if (resize_shorter_) {
      // resize_shorter set
      int shorter_side_size = spec_.GetArgument<float>("resize_shorter", ws, index);
      if (meta.H < meta.W) {
        float scale = shorter_side_size/static_cast<float>(meta.H);
        meta.rsz_h = shorter_side_size;
        meta.rsz_w = scale * meta.W;
      } else {
        float scale = shorter_side_size/static_cast<float>(meta.W);
        meta.rsz_h = scale * meta.H;
        meta.rsz_w = shorter_side_size;
      }
    } else {
      if (resize_x_) {
        if (resize_y_) {
          // resize_x and resize_y set
          meta.rsz_h = spec_.GetArgument<float>("resize_y", ws, index);
          meta.rsz_w = spec_.GetArgument<float>("resize_x", ws, index);
        } else {
          // resize_x set only
          meta.rsz_w = spec_.GetArgument<float>("resize_x", ws, index);
          float scale = static_cast<float>(meta.rsz_w) / meta.W;
          meta.rsz_h = scale * meta.H;
        }
      } else {
        // resize_y set only
        meta.rsz_h = spec_.GetArgument<float>("resize_y", ws, index);
        float scale = static_cast<float>(meta.rsz_h) / meta.H;
        meta.rsz_w = scale * meta.W;
      }
    }

    // Crop
    float crop_x_image_coord = spec_.GetArgument<float>("crop_pos_x", ws, index);
    float crop_y_image_coord = spec_.GetArgument<float>("crop_pos_y", ws, index);

    DALI_ENFORCE(crop_x_image_coord >= 0.f && crop_x_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_y_image_coord >= 0.f && crop_y_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");

    meta.crop_y = crop_y_image_coord * (meta.rsz_h - crop_h_);
    meta.crop_x = crop_x_image_coord * (meta.rsz_w - crop_w_);

    // Set mirror parameters
    meta.mirror = spec_.GetArgument<int>("mirror", ws, index);
    return meta;
  }

  // Resize meta-data
  bool resize_shorter_, resize_x_, resize_y_;

  // Crop meta-data
  int crop_h_, crop_w_;

  vector<vector<uint8>> tl_workspace_;
  vector<TransformMeta> per_thread_meta_;
  USE_OPERATOR_MEMBERS();
};

/**
 * Performs resize+crop+mirror using fast, backprojection ResizeCropMirror function
 */
template <typename Backend>
class FastResizeCropMirror : public ResizeCropMirror<Backend> {
 public:
  explicit inline FastResizeCropMirror(const OpSpec &spec) :
    ResizeCropMirror<Backend>(spec) {}

  virtual inline ~FastResizeCropMirror() = default;

 protected:
  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);
    DALI_ENFORCE(input.ndim() == 3);
    DALI_ENFORCE(IsType<uint8>(input.type()),
        "Expects input data in uint8.");
    DALI_ENFORCE(input.dim(2) == 1 || input.dim(2) == 3,
        "FastResizeCropMirror supports hwc rgb & grayscale inputs.");

    typename ResizeCropMirror<CPUBackend>::TransformMeta meta =
        ResizeCropMirror<Backend>::per_thread_meta_[ws->thread_idx()];

    // Resize the output & run
    output->Resize({crop_h_, crop_w_, meta.C});
    tl_workspace_[ws->thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    DALI_CALL(FastResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        DALI_INTERP_LINEAR,
        tl_workspace_[ws->thread_idx()].data()));
  }

  using ResizeCropMirror<Backend>::tl_workspace_;
  using ResizeCropMirror<Backend>::crop_h_;
  using ResizeCropMirror<Backend>::crop_w_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_
