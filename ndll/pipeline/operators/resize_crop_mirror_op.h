#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_

#include "ndll/pipeline/image/transform.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

// TODO(tgale): We need the batch size so that we can cache the parameters for each image
// Where should we set this? Set in the pipeline on construction and pass into all operators?
// This should work, the constructors for ops are going to be very complicated but we
// can make a macro that creates a simplified interface function for them and handles
// the passing of meta-data from the pipeline to the operator It'll need to be something
// like SetPrefetch() so that we can specify where it goes. OR we could move this meta-data
// to be set up at a later point, this kind of goes against RAII and would kinda require us
// to check to make sure we have all this stuff before we run (in DEBUG mode at least).
// This is a cleaner solution, avoids weird phony interface plus we can just make these
// methods part of the Operator base and access them through the base class. The operator class
// can verify that these are set (in DEBUG mode) in the two 'Run()' functions

template <typename Backend>
class ResizeCropMirrorOp : public Transformer<Backend> {
public:
  inline ResizeCropMirrorOp(
      int num_threads,
      std::shared_ptr<StreamPool> stream_pool,
      bool random_resize,
      bool warp_resize,
      int resize_a,
      int resize_b,
      bool random_crop,
      int crop_h,
      int crop_w,
      float mirror_prob)
    : Transformer<Backend>(num_threads, stream_pool),
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

  // This op forwards the data and writes it to files
  inline void RunPerDatumCPU(const Datum<Backend> &input, Datum<Backend> *output) override {
  }
  
  inline vector<Index> InferOutputShapeFromShape(const vector<Index> &input_shape) override {

    // Resize Options:
    // 1. Random resize (a, b), non-warping
    // 2. Random resize (a, b), warping
    // 3. Fixed resize minsize to a, non-warping
    //    - can be done w/ non-warping random resize w/ a = b
    // 3. Fixed resize to (h, w), warping

    // Crop Options:
    // 1. Center crop (h, w)
    // 2. Random crop (h, w)

    // Mirror Options:
    // 1. Mirror probability
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline ResizeCropMirrorOp* Clone() const override {
    return new ResizeCropMirrorOp(num_threads_, stream_pool_);
  }

  inline string name() const override {
    return "Dump Image Op";
  }
  
protected:
  // Resize meta-data
  int resize_a_, resize_b_;
  bool warp_resize_;
  bool random_resize_;
  
  // Crop meta-data
  bool random_crop_;
  int crop_h_, crop_w_;

  // Mirror meta-data
  float mirror_prob_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_pool_;
};

}

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_CROP_MIRROR_OP_H_
