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

namespace dali {

enum t_idInfo : uint32_t {
  t_crop = 1,
  t_mirrorHor,
  t_mirrorVert
};

struct TransformMeta {
  int H, W, C;
  int rsz_h, rsz_w;
  std::pair<int, int> crop;
  int mirror;
};


/**
 * @brief Stores parameters for resize+crop+mirror
 */
class ResizeCropMirrorAttr : protected CropAttr {
 protected:
  explicit inline ResizeCropMirrorAttr(const OpSpec &spec) : CropAttr(spec),
    interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    resize_shorter_ = spec.ArgumentDefined("resize_shorter");
    resize_longer_ = spec.ArgumentDefined("resize_longer");
    resize_x_ = spec.ArgumentDefined("resize_x");
    resize_y_ = spec.ArgumentDefined("resize_y");
    DALI_ENFORCE(!(resize_shorter_ && resize_longer_),
                 "Options `resize_longer` and `resize_shorter` are mutually"
                 " exclusive for schema \"" + spec.name() + "\"");
    DALI_ENFORCE((resize_shorter_ || resize_longer_) != (resize_x_ || resize_y_),
                 "Options `resize_{shorter,longer}` and `resize_x` or `resize_y` "
                 "are mutually exclusive for schema \"" + spec.name() + "\"");

    max_size_enforced_ = spec.ArgumentDefined("max_size");
    if (max_size_enforced_) {
      GetSingleOrRepeatedArg(spec, max_size_, "max_size", 2);
      DALI_ENFORCE(max_size_.size() > 0 && max_size_.size() <= 2,
                   "max_size has to be either a scalar or a size 2 array.");
    }
  }

 protected:
  inline const TransformMeta GetTransformMeta(const OpSpec &spec,
                                              const TensorShape<> &input_shape,
                                              const ArgumentWorkspace *ws, const Index index,
                                              const uint32_t flag = 0) {
    TransformMeta meta = {};
    meta.H = input_shape[0];
    meta.W = input_shape[1];
    meta.C = input_shape[2];

    if (resize_shorter_) {
      // resize_shorter set
      const int shorter_side_size = spec.GetArgument<float>("resize_shorter", ws, index);

      if (meta.H < meta.W) {
        const float scale = shorter_side_size / static_cast<float>(meta.H);
        meta.rsz_h = shorter_side_size;
        meta.rsz_w = static_cast<int>(std::round(scale * meta.W));
        if (max_size_enforced_) {
          if (meta.rsz_w > max_size_[1]) {
            const float ratio = static_cast<float>(meta.H) / static_cast<float>(meta.W);
            meta.rsz_h = static_cast<int>(std::round(ratio * max_size_[1]));
            meta.rsz_w = max_size_[1];
          }
        }
      } else {
        const float scale = shorter_side_size / static_cast<float>(meta.W);
        meta.rsz_h = static_cast<int>(std::round(scale * meta.H));
        meta.rsz_w = shorter_side_size;
        if (max_size_enforced_) {
          if (meta.rsz_h > max_size_[0]) {
            const float ratio = static_cast<float>(meta.W) / static_cast<float>(meta.H);
            meta.rsz_h = max_size_[0];
            meta.rsz_w = static_cast<int>(std::round(ratio * max_size_[0]));
          }
        }
      }
    } else if (resize_longer_) {
        // resize_longer set
        const int longer_side_size = spec.GetArgument<float>("resize_longer", ws, index);

        if (meta.H > meta.W) {
          const float scale = longer_side_size / static_cast<float>(meta.H);
          meta.rsz_h = longer_side_size;
          meta.rsz_w = static_cast<int>(std::round(scale * meta.W));
        } else {
          const float scale = longer_side_size / static_cast<float>(meta.W);
          meta.rsz_h = static_cast<int>(std::round(scale * meta.H));
          meta.rsz_w = longer_side_size;
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
          meta.rsz_h = static_cast<int>(std::round(scale * meta.H));
        }
      } else {
        // resize_y set only
        meta.rsz_h = spec.GetArgument<float>("resize_y", ws, index);
        const float scale = static_cast<float>(meta.rsz_h) / meta.H;
        meta.rsz_w = static_cast<int>(std::round(scale * meta.W));
      }
    }

    if (flag & t_crop) {
      float crop_anchor_norm[2];
      crop_anchor_norm[0] = spec.GetArgument<float>("crop_pos_y", ws, index);
      crop_anchor_norm[1] = spec.GetArgument<float>("crop_pos_x", ws, index);

      auto anchor_abs = CalculateAnchor(make_span(crop_anchor_norm),
                                        {crop_height_[index], crop_width_[index]},
                                        {meta.rsz_h, meta.rsz_w});
      meta.crop = {anchor_abs[0], anchor_abs[1]};
    }

    if (flag & t_mirrorHor) {
      // Set mirror parameters
      meta.mirror = spec.GetArgument<int>("mirror", ws, index);
    }

    return meta;
  }

  /**
   * @brief Enforce that all shapes match
   *
   * @param ws
   * @return const vector<Index> One matching shape for all inputs
   */
  virtual const std::vector<Index> CheckShapes(const SampleWorkspace *ws) {
    const auto &input = ws->Input<CPUBackend>(0);
    // enforce that all shapes match
    for (int i = 1; i < ws->NumInput(); ++i) {
      DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }

    DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");

    return std::vector<Index>{input.shape().begin(), input.shape().end()};
  }

  inline const TransformMeta GetTransfomMeta(const SampleWorkspace *ws, const OpSpec &spec) {
    const auto input_shape = CheckShapes(ws);
    return GetTransformMeta(spec, input_shape, ws, ws->data_idx(), ResizeInfoNeeded());
  }

  DALIInterpType getInterpType() const        { return interp_type_; }
  virtual uint32_t ResizeInfoNeeded() const       { return t_crop + t_mirrorHor; }

  // Interpolation type
  DALIInterpType interp_type_;

 private:
  // Resize meta-data
  bool resize_shorter_, resize_longer_, resize_x_, resize_y_;

  bool max_size_enforced_;
  // Contains (H, W) max sizes
  std::vector<float> max_size_;
};

typedef DALIError_t (*resizeCropMirroHost)(const uint8 *img, int H, int W, int C,
                                 int rsz_h, int rsz_w, const std::pair<int, int> &crop, int crop_h,
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
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    return false;
  }

  inline void SetupSharedSampleParams(SampleWorkspace &ws) override {
    per_thread_meta_[ws.thread_idx()] = GetTransfomMeta(&ws, spec_);
  }

  inline void RunImpl(SampleWorkspace &ws) override {
    RunResizeImpl(ws, ResizeCropMirrorHost);
  }

  inline void RunResizeImpl(SampleWorkspace &ws, resizeCropMirroHost func) {
    auto &input = ws.Input<CPUBackend>(0);
    auto &output = ws.Output<CPUBackend>(0);
    CheckParam(input, "ResizeCropMirror");

    const TransformMeta &meta = per_thread_meta_[ws.thread_idx()];

    // Resize the output & run
    output.Resize(
        std::vector<Index>{crop_height_[ws.data_idx()], crop_width_[ws.data_idx()], meta.C});
    output.SetLayout(input.GetLayout());

    tl_workspace_[ws.thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    DALI_CALL((*func)(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop,
        crop_height_[0], crop_width_[0],
        meta.mirror,
        output.template mutable_data<uint8>(),
        interp_type_,
        tl_workspace_[ws.thread_idx()].data()));
  }

  vector<vector<uint8>> tl_workspace_;
  vector<TransformMeta> per_thread_meta_;
  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;
  using Operator<Backend>::SetupSharedSampleParams;
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
  inline void RunImpl(SampleWorkspace &ws) override {
    RunResizeImpl(ws, FastResizeCropMirrorHost);
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_CROP_MIRROR_H_
