#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_

#include <random>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/transformer.h"

namespace ndll {

template <typename Backend>
class ResizeOp : public Transformer<Backend> {
public:
  inline ResizeOp(const OpSpec &spec) :
    Transformer<Backend>(spec),
    rand_gen_(time(nullptr)),
    random_resize_(spec.GetSingleArgument<bool>("random_resize", false)),
    warp_resize_(spec.GetSingleArgument<bool>("warp_resize", false)),
    resize_a_(spec.GetSingleArgument<int>("resize_a", -1)),
    resize_b_(spec.GetSingleArgument<int>("resize_b", -1)),
    image_type_(spec.GetSingleArgument<NDLLImageType>("image_type", NDLL_RGB)),
    color_(IsColor(image_type_)), C_(color_ ? 3 : 1),
    type_(spec.GetSingleArgument<NDLLInterpType>("interp_type", NDLL_INTERP_LINEAR)) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a_ > 0 && resize_b_ > 0);
    NDLL_ENFORCE(resize_a_ <= resize_b_);

    // Resize per-image data
    input_ptrs_.resize(batch_size_);
    output_ptrs_.resize(batch_size_);
    input_sizes_.resize(batch_size_);
    output_sizes_.resize(batch_size_);    
  }
    
  virtual inline ~ResizeOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int /* unused */) override {
    NDLL_ENFORCE(input_shape.size() == 3, "Resize requires 3-dim image");
    NDLL_ENFORCE(input_shape[2] == C_,
        "Input channel dimension does not match the number of " 
        "channels set in the op constructor: " +
        std::to_string(input_shape[2]) + " v. " + std::to_string(C_));
    NDLLSize &in_size = input_sizes_[data_idx];
    in_size.height = input_shape[0];
    in_size.width = input_shape[1];
      
    NDLLSize &out_size = output_sizes_[data_idx];
    if (random_resize_ && warp_resize_) {
      // random resize + warp. Select a new size for both dims of
      // the image uniformly from the range [resize_a_, resize_b_]
      out_size.height = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
      out_size.width = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
    } else if (random_resize_) {
      // random + no warp. We select a new size of the smallest side
      // of the image uniformly in the range [resize_a_, resize_b_]
      if (in_size.width < in_size.height) {
        out_size.width = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
        out_size.height = static_cast<float>(in_size.height) / in_size.width * out_size.width;
      } else {
        out_size.height = std::uniform_int_distribution<>(resize_a_, resize_b_)(rand_gen_);
        out_size.width = static_cast<float>(in_size.width) / in_size.height * out_size.height;
      }
    } else if (warp_resize_) {
      // no random + warp. We take the new dims to be h = resize_a_
      // and w = resize_b_
      out_size.height = resize_a_;
      out_size.width = resize_b_;
    } else { 
      // no random + no warp. In this mode resize_b_ is ignored and
      // the input image is resizes such that the smallest side is
      // >= resize_a_
      if (in_size.width < in_size.height) {
        if (in_size.width < resize_a_) {
          out_size.width = resize_a_;
          out_size.height = static_cast<float>(in_size.height) / in_size.width * out_size.width;
        }
      } else {
        if (in_size.height < resize_a_) {
          out_size.height = resize_a_;
          out_size.width = static_cast<float>(in_size.width) / in_size.height * out_size.height;
        }
      }
    }

    return std::vector<Index>{out_size.height, out_size.width, C_};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template mutable_data<uint8>();
  }
  
  inline string name() const override {
    return "ResizeOp";
  }

protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    BatchedResize((const uint8**)input_ptrs_.data(), batch_size_, C_, input_sizes_.data(),
        output_ptrs_.data(), output_sizes_.data(), type_);
  }
  
  inline void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend> *output, int data_idx, int /* unused */) override {
    // Setup input & output ptrs for each image
    input_ptrs_[data_idx] = input.template sample<uint8>(data_idx);
    output_ptrs_[data_idx] = output->template mutable_sample<uint8>(data_idx);
  }
  
  std::mt19937 rand_gen_;
  
  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;

  // Input/output channels meta-data
  NDLLImageType image_type_;
  bool color_;
  int C_;

  // Interpolation type
  NDLLInterpType type_;
  
  vector<const uint8*> input_ptrs_;
  vector<uint8*> output_ptrs_;
  
  vector<NDLLSize> input_sizes_, output_sizes_;
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_
