#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_

#include <random>
#include <type_traits>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/transformer.h"

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
class ResizeCropMirrorOp : public Transformer<Backend> {
public:
  inline ResizeCropMirrorOp(const OpSpec &spec) :
    Transformer<Backend>(spec),
    rand_gen_(time(nullptr)),
    random_resize_(spec.GetSingleArgument<bool>("random_resize", false)),
    warp_resize_(spec.GetSingleArgument<bool>("warp_resize", false)),
    resize_a_(spec.GetSingleArgument<int>("resize_a", -1)),
    resize_b_(spec.GetSingleArgument<int>("resize_b", -1)),
    random_crop_(spec.GetSingleArgument<bool>("random_crop", false)),
    crop_h_(spec.GetSingleArgument<int>("crop_h", -1)),
    crop_w_(spec.GetSingleArgument<int>("crop_w", -1)),
    mirror_prob_(spec.GetSingleArgument<float>("mirror_prob", 0.5f)) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a_ > 0 && resize_b_ > 0);
    NDLL_ENFORCE(resize_a_ <= resize_b_);
    NDLL_ENFORCE(crop_h_ > 0 && crop_w_ > 0);
    NDLL_ENFORCE(mirror_prob_ <= 1.f && mirror_prob_ >= 0.f);

    // Resize per-image & per-thread data
    tl_workspace_.resize(num_threads_);
    transform_params_.resize(batch_size_);
  }

  virtual inline ~ResizeCropMirrorOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int /* unused */) override {
#ifndef NDEBUG
    NDLL_ENFORCE(data_idx < batch_size_, "data_idx out of range");
#endif
    NDLL_ENFORCE(input_shape.size() == 3, "ResizeCropMirror requires 3-dim image");
    NDLL_ENFORCE(input_shape[2] == 1 || input_shape[2] == 3,
        "ResizeCropMirror supports HWC rgb & grayscale images");
    TransformMeta &meta = transform_params_[data_idx];
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
        if (meta.W < resize_a_) {
          meta.rsz_w = resize_a_;
          meta.rsz_h = static_cast<float>(meta.H) / meta.W * meta.rsz_w;
        }
      } else {
        if (meta.H < resize_a_) {
          meta.rsz_h = resize_a_;
          meta.rsz_w = static_cast<float>(meta.W) / meta.H * meta.rsz_h;
        }
      }
    }

    // Set crop parameters
    if (random_crop_) {
      meta.crop_y = std::uniform_int_distribution<>(0, meta.rsz_h - crop_h_)(rand_gen_);
      meta.crop_x = std::uniform_int_distribution<>(0, meta.rsz_w - crop_w_)(rand_gen_);
    } else {
      meta.crop_y = (meta.rsz_h - crop_h_) / 2;
      meta.crop_x = (meta.rsz_w - crop_w_) / 2;
    }

    // Set mirror parameters
    meta.mirror = std::bernoulli_distribution(mirror_prob_)(rand_gen_);
    return std::vector<Index>{crop_h_, crop_w_, meta.C};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template mutable_data<uint8>();
  }
  
  inline string name() const override {
    return "ResizeCropMirrorOp";
  }

protected:
   inline void RunPerSampleCPU(const Sample<Backend> &input,
      Sample<Backend> *output, int data_idx, int thread_idx) override {
    TransformMeta &meta = transform_params_[data_idx];
#ifndef NDEBUG
    NDLL_ENFORCE(input.shape().size() == 3);
    NDLL_ENFORCE(input.shape()[0] == meta.H);
    NDLL_ENFORCE(input.shape()[1] == meta.W);
    NDLL_ENFORCE(input.shape()[2] == meta.C);
    NDLL_ENFORCE(output->shape().size() == 3);
    NDLL_ENFORCE(output->shape()[0] == crop_h_);
    NDLL_ENFORCE(output->shape()[1] == crop_w_);
    NDLL_ENFORCE(output->shape()[2] == meta.C);
#endif
    tl_workspace_[thread_idx].resize(meta.rsz_h*meta.rsz_w*meta.C);
    ResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        NDLL_INTERP_LINEAR,
        tl_workspace_[thread_idx].data());
  }
  
  std::mt19937 rand_gen_;
  
  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;
  
  // Crop meta-data
  bool random_crop_;
  int crop_h_, crop_w_;

  // Mirror meta-data
  float mirror_prob_;

  struct TransformMeta {
    int H, W, C;
    int rsz_h, rsz_w;
    int crop_x, crop_y;
    bool mirror;
  };
  vector<TransformMeta> transform_params_;
  vector<vector<uint8>> tl_workspace_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
};

/**
 * Performs resize+crop+mirror using fast, backprojection ResizeCropMirror function
 */ 
template <typename Backend>
class FastResizeCropMirrorOp : public ResizeCropMirrorOp<Backend> {
public:
  inline FastResizeCropMirrorOp(const OpSpec &spec) :
    ResizeCropMirrorOp<Backend>(spec) {}

  virtual inline ~FastResizeCropMirrorOp() = default;

  inline string name() const override {
    return "FastResizeCropMirrorOp";
  }
  
protected:
    inline void RunPerSampleCPU(const Sample<Backend> &input,
      Sample<Backend> *output, int data_idx, int thread_idx) override {
    typename ResizeCropMirrorOp<Backend>::TransformMeta &meta =
      transform_params_[data_idx];
#ifndef NDEBUG
    NDLL_ENFORCE(input.shape().size() == 3);
    NDLL_ENFORCE(input.shape()[0] == meta.H);
    NDLL_ENFORCE(input.shape()[1] == meta.W);
    NDLL_ENFORCE(input.shape()[2] == meta.C);
    NDLL_ENFORCE(output->shape().size() == 3);
    NDLL_ENFORCE(output->shape()[0] == crop_h_);
    NDLL_ENFORCE(output->shape()[1] == crop_w_);
    NDLL_ENFORCE(output->shape()[2] == meta.C);
#endif
    tl_workspace_[thread_idx].resize(crop_h_*crop_w_*meta.C);
    FastResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template mutable_data<uint8>(),
        NDLL_INTERP_LINEAR,
        tl_workspace_[thread_idx].data());
  }
  
  using ResizeCropMirrorOp<Backend>::random_resize_;
  using ResizeCropMirrorOp<Backend>::warp_resize_;
  using ResizeCropMirrorOp<Backend>::resize_a_;
  using ResizeCropMirrorOp<Backend>::resize_b_;
  using ResizeCropMirrorOp<Backend>::random_crop_;
  using ResizeCropMirrorOp<Backend>::crop_h_;
  using ResizeCropMirrorOp<Backend>::crop_w_;
  using ResizeCropMirrorOp<Backend>::mirror_prob_;
  using ResizeCropMirrorOp<Backend>::transform_params_;
  using ResizeCropMirrorOp<Backend>::tl_workspace_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
