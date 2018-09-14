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
#include "dali/pipeline/operators/common.h"

namespace dali {

typedef enum {
  t_crop = 1,
  t_mirrorHor,
  t_mirrorVert
} t_idInfo;

/**
 * @brief Stores parameters for resize+crop+mirror
 */
class ResizeCropMirrorAttr {
 protected:
  explicit inline ResizeCropMirrorAttr(const OpSpec &spec) :
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    resize_shorter_ = spec.ArgumentDefined("resize_shorter");
    resize_x_ = spec.ArgumentDefined("resize_x");
    resize_y_ = spec.ArgumentDefined("resize_y");
    DALI_ENFORCE(resize_shorter_ != (resize_x_ || resize_y_),
                 "Options `resize_shorter` and `resize_x` or `resize_y` "
                 "are mutually exclusive for schema \"" + spec.name() + "\"");

    if (spec.name() != "Resize") {
      vector<int>cropTmp;
      GetSingleOrRepeatedArg(spec, &cropTmp, "crop", 2);
      crop_[0] = cropTmp[0];
      crop_[1] = cropTmp[1];
      DALI_ENFORCE(crop_[0] > 0 && crop_[1] > 0);
    }
  }

  struct TransformMeta {
    int H, W, C;
    int rsz_h, rsz_w;
    int crop_x, crop_y;
    int mirror;
  };

 protected:
  inline const TransformMeta GetTransformMeta(const OpSpec &spec, const vector<Index> &input_shape,
                        const ArgumentWorkspace *ws, const Index index, const uint flag = 0) {
    TransformMeta meta;
    meta.H = input_shape[0];
    meta.W = input_shape[1];
    meta.C = input_shape[2];

    if (resize_shorter_) {
      // resize_shorter set
      const int shorter_side_size = spec.GetArgument<float>("resize_shorter", ws, index);
      if (meta.H < meta.W) {
        const float scale = shorter_side_size/static_cast<float>(meta.H);
        meta.rsz_h = shorter_side_size;
        meta.rsz_w = scale * meta.W;
      } else {
        const float scale = shorter_side_size/static_cast<float>(meta.W);
        meta.rsz_h = scale * meta.H;
        meta.rsz_w = shorter_side_size;
      }
    } else {
      if (resize_x_) {
        meta.rsz_w = spec.GetArgument<float>("resize_x", ws, index);
        if (resize_y_) {
          // resize_x and resize_y set
          meta.rsz_h = spec.GetArgument<float>("resize_y", ws, index);
        } else {
          // resize_x set only
          const float scale = static_cast<float>(meta.rsz_w) / meta.W;
          meta.rsz_h = scale * meta.H;
        }
      } else {
        // resize_y set only
        meta.rsz_h = spec.GetArgument<float>("resize_y", ws, index);
        const float scale = static_cast<float>(meta.rsz_h) / meta.H;
        meta.rsz_w = scale * meta.W;
      }
    }

    if (flag & t_crop) {
      // Crop
      const auto crop_x_image_coord = spec.GetArgument<float>("crop_pos_x", ws, index);
      const auto crop_y_image_coord = spec.GetArgument<float>("crop_pos_y", ws, index);

      DALI_ENFORCE(crop_x_image_coord >= 0.f && crop_x_image_coord <= 1.f,
                   "Crop coordinates need to be in range [0.0, 1.0]");
      DALI_ENFORCE(crop_y_image_coord >= 0.f && crop_y_image_coord <= 1.f,
                   "Crop coordinates need to be in range [0.0, 1.0]");

      meta.crop_y = crop_y_image_coord * (meta.rsz_h - crop_[0]);
      meta.crop_x = crop_x_image_coord * (meta.rsz_w - crop_[1]);
    }

    if (flag & t_mirrorHor) {
      // Set mirror parameters
      meta.mirror = spec.GetArgument<int>("mirror", ws, index);
    }

    return meta;
  }

  inline const TransformMeta GetTransfomMeta(const SampleWorkspace *ws, const OpSpec &spec) {
    auto &input = ws->Input<CPUBackend>(0);

    // enforce that all shapes match
    for (int i = 1; i < ws->NumInput(); ++i) {
      DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }

    return GetTransformMeta(spec, input.shape(), ws, ws->data_idx(), ResizeInfoNeeded());
  }

  DALIInterpType getInterpType() const        { return interp_type_; }
  virtual uint ResizeInfoNeeded() const       { return t_crop + t_mirrorHor; }

  DALIImageType image_type_;
  // Interpolation type
  DALIInterpType interp_type_;
  // Crop meta-data
  array<int, 2>crop_ = {{0}};

 private:
  // Resize meta-data
  bool resize_shorter_, resize_x_, resize_y_;
};

typedef DALIError_t (*resizeCropMirroHost)(const uint8 *img, int H, int W, int C,
                                 int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
                                 int crop_w, int mirror, uint8 *out_img, DALIInterpType type,
                                 uint8 *workspace);
/**
 * @brief Performs fused resize+crop+mirror
 */
template <typename Backend>
class ResizeCropMirror : public Operator<CPUBackend>, protected ResizeCropMirrorAttr {
 public:
  explicit inline ResizeCropMirror(const OpSpec &spec) :
    Operator(spec), ResizeCropMirrorAttr(spec) {
    // Resize per-image & per-thread data
    tl_workspace_.resize(num_threads_);

    // per-image-set data
    per_thread_meta_.resize(num_threads_);
  }

  ~ResizeCropMirror() override = default;

 protected:
  inline void SetupSharedSampleParams(SampleWorkspace *ws) override {
    per_thread_meta_[ws->thread_idx()] = GetTransfomMeta(ws, spec_);
  }

  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    RunResizeImpl(ws, idx, ResizeCropMirrorHost);
  }

  inline void RunResizeImpl(SampleWorkspace *ws, const int idx, resizeCropMirroHost func) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);
    CheckParam(input, "ResizeCropMirror");

    const TransformMeta &meta = per_thread_meta_[ws->thread_idx()];

    // Resize the output & run
    output->Resize({crop_[0], crop_[1], meta.C});

    tl_workspace_[ws->thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    DALI_CALL((*func)(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_[0], crop_[1],
        meta.mirror,
        output->template mutable_data<uint8>(),
        interp_type_,
        tl_workspace_[ws->thread_idx()].data()));
  }

  vector<vector<uint8>> tl_workspace_;
  vector<TransformMeta> per_thread_meta_;
  USE_OPERATOR_MEMBERS();
};

/**
 * Performs resize+crop+mirror using fast, backprojection ResizeCropMirror function
 */
template <typename Backend>
class FastResizeCropMirror : public ResizeCropMirror<CPUBackend> {
 public:
  explicit inline FastResizeCropMirror(const OpSpec &spec) :
    ResizeCropMirror<CPUBackend>(spec) {}

  inline ~FastResizeCropMirror() override = default;

 protected:
  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    RunResizeImpl(ws, idx, FastResizeCropMirrorHost);
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_
