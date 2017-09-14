#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_

#include <random>

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
class ResizeCropMirrorOp : public Transformer<Backend> {
public:
  inline ResizeCropMirrorOp(
      bool random_resize,
      bool warp_resize,
      int resize_a,
      int resize_b,
      bool random_crop,
      int crop_h,
      int crop_w,
      float mirror_prob)
    : rand_gen_(time(nullptr)),
    random_resize_(random_resize),
    warp_resize_(warp_resize),
    resize_a_(resize_a),
    resize_b_(resize_b),
    random_crop_(random_crop),
    crop_h_(crop_h),
    crop_w_(crop_w),
    mirror_prob_(mirror_prob) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a > 0 && resize_b > 0);
    NDLL_ENFORCE(resize_a <= resize_b);
    NDLL_ENFORCE(crop_h > 0 && crop_w > 0);
    NDLL_ENFORCE(mirror_prob <= 1.f && mirror_prob >= 0.f);
  }

  virtual inline ~ResizeCropMirrorOp() = default;

  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int data_idx) override {
    TransformMeta &meta = transform_params_[data_idx];
#ifdef DEBUG
    NDLL_ENFORCE(input.shape().size() == 3);
    NDLL_ENFORCE(input.shape()[0] == meta.H);
    NDLL_ENFORCE(input.shape()[1] == meta.W);
    NDLL_ENFORCE(input.shape()[2] == meta.C);
    NDLL_ENFORCE(output->shape().size() == 3);
    NDLL_ENFORCE(output->shape()[0] == crop_h_);
    NDLL_ENFORCE(output->shape()[1] == crop_w_);
    NDLL_ENFORCE(output->shape()[2] == meta.C);
#endif
    ResizeCropMirrorHost(
        input.template data<uint8>(),
        meta.H, meta.W, meta.C,
        meta.rsz_h, meta.rsz_w,
        meta.crop_y, meta.crop_x,
        crop_h_, crop_w_,
        meta.mirror,
        output->template data<uint8>());
  }
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx) override {
    // TODO(tgale): This is kind of gross and unintuitive that we
    // can't do this in the constructor. In reality this info is
    // we just also don't want the user to have to pass in the
    // same meta-data over and over and over again. is there a
    // better way to do this? We kind of violate RAII a bit w/ this...
    // a) Macro-ed wrappers to declare entry functions that handle gross-ness
    //   - a bit opaque, but just provides a convenience layer
    //   - tough because we have to differentiate between prefetch & forward
    //     - can we do this with template specialization on the Backend? Probably...
    // b) Alternate constructors that set the desired meta-data. have the pipeline
    //    call these when it constructs the ops ... nvm, "Clone()" is called to construct
    //    the ops so the pipeline can't affect the constructor. Would have to add params
    //    to this, which would be grosser. Very much ties these classes to the pipeline
    // a seems like the best route.
    //
    // We're guaranteed that the batch size is set by now,
    // use it to resize our transform params
    if (transform_params_.size() == 0) transform_params_.resize(batch_size_);
#ifdef DEBUG
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
    
    // DEBUG
    // cout << "resize dims: " << meta.rsz_h << "x" << meta.rsz_w << endl;
    // cout << "crop dims: " << crop_h_ << "x" << crop_w_ << endl;
    // cout << "crop offset: " << meta.crop_y << " " << meta.crop_x << endl;
    // cout << "mirror: " << meta.mirror << endl;

    return std::vector<Index>{crop_h_, crop_w_, meta.C};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline ResizeCropMirrorOp* Clone() const override {
    return new ResizeCropMirrorOp(random_resize_, warp_resize_, resize_a_,
        resize_b_, random_crop_, crop_h_, crop_w_, mirror_prob_);
  }

  inline string name() const override {
    return "ResizeCropMirrorOp";
  }
  
protected:
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
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

}

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
