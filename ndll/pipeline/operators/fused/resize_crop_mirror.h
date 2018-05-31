// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_
#define NDLL_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_

#include <random>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

/**
 * @brief Performs fused resize+crop+mirror
 *
 * Resize Options:
 * 1. Random resize (a, b), non-warping
 * 2. Random resize (a, b), warping
 * 3. Fixed resize minsize to a, non-warping
 *    - can be done w/ non-warping random resize w/ a = b
 * 3. Fixed resize to (h, w), warping
 *
 * Crop Options:
 * 1. Center crop (h, w)
 * 2. Random crop (h, w)
 *
 * Mirror Options:
 * 1. Mirror probability
 */

template <typename Backend>
class ResizeCropMirror : public Operator<CPUBackend> {
 public:
  explicit inline ResizeCropMirror(const OpSpec &spec) :
    Operator(spec),
    random_resize_(spec.GetArgument<bool>("random_resize")),
    warp_resize_(spec.GetArgument<bool>("warp_resize")),
    resize_a_(spec.GetArgument<int>("resize_a")),
    resize_b_(spec.GetArgument<int>("resize_b")) {
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
        NDLL_FAIL("Invalid type of argument \"crop\". Expected int or list of int");
      }
    }

    NDLL_ENFORCE(temp_crop.size() == 2, "Argument \"crop\" expects a list of at most 2 elements, "
        + to_string(temp_crop.size()) + " given.");
    crop_h_ = temp_crop[0];
    crop_w_ = temp_crop[1];
    // Validate input parameters
    NDLL_ENFORCE(resize_a_ > 0 && resize_b_ > 0);
    NDLL_ENFORCE(resize_a_ <= resize_b_);
    NDLL_ENFORCE(crop_h_ > 0 && crop_w_ > 0);

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
      NDLL_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }

    per_thread_meta_[ws->thread_idx()] = GetTransformMeta(input.shape(), ws, ws->data_idx());
  }

  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);
    NDLL_ENFORCE(input.ndim() == 3);
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Expects input data in uint8.");
    NDLL_ENFORCE(input.dim(2) == 1 || input.dim(2) == 3,
        "ResizeCropMirror supports hwc rgb & grayscale inputs.");

    const TransformMeta &meta = per_thread_meta_[ws->thread_idx()];

    // Resize the output & run
    output->Resize({crop_h_, crop_w_, meta.C});

    tl_workspace_[ws->thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    NDLL_CALL(ResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        NDLL_INTERP_LINEAR,
        tl_workspace_[ws->thread_idx()].data()));
  }

  inline const TransformMeta GetTransformMeta(const vector<Index> &input_shape,
                                              SampleWorkspace * ws, const Index index) {
    TransformMeta meta;
    meta.H = input_shape[0];
    meta.W = input_shape[1];
    meta.C = input_shape[2];

    if (random_resize_ && warp_resize_) {
      // random resize + warp. Select a new size for both dims of
      // the image uniformly from the range [resize_a_, resize_b_]
      meta.rsz_h = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
      meta.rsz_w = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
    } else if (random_resize_) {
      // random + no warp. We select a new size of the smallest side
      // of the image uniformly in the range [resize_a_, resize_b_]
      if (meta.W < meta.H) {
        meta.rsz_w = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
        meta.rsz_h = static_cast<float>(meta.H) / meta.W * meta.rsz_w;
      } else {
        meta.rsz_h = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
        meta.rsz_w = static_cast<float>(meta.W) / meta.H * meta.rsz_h;
      }
    } else if (warp_resize_) {
      // no random + warp. We take the new dims to be h = resize_a_
      // and w = resize_b_
      meta.rsz_h = resize_a_;
      meta.rsz_w = resize_b_;
    } else {
      // no random + no warp. In this mode resize_b_ is ignored and
      // the input image is resizes such that the smallest side is
      // >= resize_a_
      if (meta.W < meta.H) {
          meta.rsz_w = resize_a_;
          meta.rsz_h = static_cast<float>(meta.H) / meta.W * meta.rsz_w;
      } else {
          meta.rsz_h = resize_a_;
          meta.rsz_w = static_cast<float>(meta.W) / meta.H * meta.rsz_h;
      }
    }

    // Crop
    float crop_x_image_coord = spec_.GetArgument<float>("crop_pos_x", ws, index);
    float crop_y_image_coord = spec_.GetArgument<float>("crop_pos_y", ws, index);

    NDLL_ENFORCE(crop_x_image_coord >= 0.f && crop_x_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");
    NDLL_ENFORCE(crop_y_image_coord >= 0.f && crop_y_image_coord <= 1.f,
        "Crop coordinates need to be in range [0.0, 1.0]");

    meta.crop_y = crop_y_image_coord * (meta.rsz_h - crop_h_);
    meta.crop_x = crop_x_image_coord * (meta.rsz_w - crop_w_);

    // Set mirror parameters
    meta.mirror = spec_.GetArgument<int>("mirror", ws, index);
    return meta;
  }

  std::mt19937 rand_gen_;

  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;

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
    NDLL_ENFORCE(input.ndim() == 3);
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Expects input data in uint8.");
    NDLL_ENFORCE(input.dim(2) == 1 || input.dim(2) == 3,
        "FastResizeCropMirror supports hwc rgb & grayscale inputs.");

    typename ResizeCropMirror<CPUBackend>::TransformMeta meta =
        ResizeCropMirror<Backend>::per_thread_meta_[ws->thread_idx()];

    // Resize the output & run
    output->Resize({crop_h_, crop_w_, meta.C});
    tl_workspace_[ws->thread_idx()].resize(meta.rsz_h*meta.rsz_w*meta.C);
    NDLL_CALL(FastResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        NDLL_INTERP_LINEAR,
        tl_workspace_[ws->thread_idx()].data()));
  }

  using ResizeCropMirror<Backend>::tl_workspace_;
  using ResizeCropMirror<Backend>::crop_h_;
  using ResizeCropMirror<Backend>::crop_w_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_FUSED_RESIZE_CROP_MIRROR_H_
