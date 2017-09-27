#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend>
class ResizeOp : public Transformer<Backend> {
public:
  inline ResizeOp(
      bool random_resize,
      bool warp_resize,
      int resize_a,
      int resize_b,
      bool color,
      NDLLInterpType type = NDLL_INTERP_LINEAR)
    : rand_gen_(time(nullptr)),
      random_resize_(random_resize),
      warp_resize_(warp_resize),
      resize_a_(resize_a),
      resize_b_(resize_b),
      color_(color),
      C_(color_ ? 3 : 1),
      type_(type) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a > 0 && resize_b > 0);
    NDLL_ENFORCE(resize_a <= resize_b);
  }

  virtual inline ~ResizeOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int /* unused */) override {
#ifndef NDEBUG
    NDLL_ENFORCE(data_idx < batch_size_, "data_idx out of range");
#endif
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
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<uint8>();
  }
  
  inline ResizeOp* Clone() const override {
    return new ResizeOp(random_resize_, warp_resize_, resize_a_, resize_b_, color_, type_);
  }

  inline string name() const override {
    return "ResizeOp";
  }

  inline void set_num_threads(int num_threads) override {
    num_threads_ = num_threads;
  }

  // User can override if they need to setup meta-data
  inline void set_batch_size(int batch_size) override {
    batch_size_ = batch_size;

    input_ptrs_.resize(batch_size);
    output_ptrs_.resize(batch_size);
    input_sizes_.resize(batch_size);
    output_sizes_.resize(batch_size);
  }
  
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    nppSetStream(stream_pool_->GetStream());
    BatchedResize((const uint8**)input_ptrs_.data(), batch_size_, C_, input_sizes_.data(),
        output_ptrs_.data(), output_sizes_.data(), type_);
  }
  
  inline void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend> *output, int data_idx, int /* unused */) override {
    // Setup input & output ptrs for each image
    input_ptrs_[data_idx] = input.template datum<uint8>(data_idx);
    output_ptrs_[data_idx] = output->template datum<uint8>(data_idx);
  }
  
  std::mt19937 rand_gen_;
  
  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;

  // Input/output channels meta-data
  bool color_;
  int C_;

  // Interpolation type
  NDLLInterpType type_;
  
  vector<const uint8*> input_ptrs_;
  vector<uint8*> output_ptrs_;
  
  vector<NDLLSize> input_sizes_, output_sizes_;
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_RESIZE_OP_H_
