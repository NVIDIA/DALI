#ifndef NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_
#define NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_

#include <cstring>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend, typename OUT>
class CropMirrorNormalizePermuteOp : public Transformer<Backend> {
public:
  inline CropMirrorNormalizePermuteOp(
      bool random_crop,
      int crop_h,
      int crop_w,
      float mirror_prob,
      bool color,
      vector<float> mean,
      vector<float> std)
    : rand_gen_(time(nullptr)),
      random_crop_(random_crop),
      crop_h_(crop_h),
      crop_w_(crop_w),
      mirror_prob_(mirror_prob),
      color_(color),
      C_(color ? 3 : 1),
      mean_vec_(mean),
      std_vec_(std) {
    // Validate input parameters
    NDLL_ENFORCE(crop_h > 0 && crop_w > 0);
    NDLL_ENFORCE(mirror_prob <= 1.f && mirror_prob >= 0.f);

    // Validate & save mean & std params
    NDLL_ENFORCE((int)mean.size() == C_);
    NDLL_ENFORCE((int)std.size() == C_);
    for (auto &val : std) {
      NDLL_ENFORCE(val != 0, "stddev must be non-zero");
    }
    mean_.Copy(mean);
    std_.Copy(std);

    // We need three buffers for our batched parameters
    // in_ptrs, in_strides, and mirror parameters per image
    batched_param_sizes_.resize(3);
  }

  virtual inline ~CropMirrorNormalizePermuteOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int /* unused */) override {
    NDLL_ENFORCE(input_shape.size() == 3, "CropMirrorNormalizePermute requires 3-dim image");
    NDLL_ENFORCE(input_shape[2] == C_,
        "Input channel dimension does not match the number of " 
        "channels set in the op constructor: " +
        std::to_string(input_shape[2]) + " v. " + std::to_string(C_));
    int H = input_shape[0];
    int W = input_shape[1];

    // Set crop parameters
    int crop_y, crop_x;
    if (random_crop_) {
      crop_y = std::uniform_int_distribution<>(0, H - crop_h_)(rand_gen_);
      crop_x = std::uniform_int_distribution<>(0, W - crop_w_)(rand_gen_);
    } else {
      crop_y = (H - crop_h_) / 2;
      crop_x = (W - crop_w_) / 2;
    }

    // Save image stride & crop offset
    in_strides_[data_idx] = W*C_;
    crop_offsets_[data_idx] = crop_y*W*C_ + crop_x*C_;

    // Set mirror parameters
    mirror_[data_idx] = std::bernoulli_distribution(mirror_prob_)(rand_gen_);
    return std::vector<Index>{C_, crop_h_, crop_w_};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<OUT>();
  }
  
  inline CropMirrorNormalizePermuteOp* Clone() const override {
    return new CropMirrorNormalizePermuteOp(random_crop_, crop_h_,
        crop_w_, mirror_prob_, color_, mean_vec_, std_vec_);
  }

  inline string name() const override {
    return "CropMirrorNormalizePermuteOp";
  }

  inline void set_num_threads(int num_threads) override {
    num_threads_ = num_threads;
  }

  // User can override if they need to setup meta-data
  inline void set_batch_size(int batch_size) override {
    batch_size_ = batch_size;

    crop_offsets_.resize(batch_size);
    in_strides_.resize(batch_size);
    mirror_.resize(batch_size);
  }
  
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    NDLL_CALL(BatchedCropMirrorNormalizePermute(
            batched_param_gpu_buffers_[0].template data<const uint8*>(),
            batched_param_gpu_buffers_[1].template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            batched_param_gpu_buffers_[2].template data<bool>(),
            mean_.template data<float>(),
            std_.template data<float>(),
            output->template data<OUT>(),
            stream_pool_->GetStream()));
  }
  
  inline void CalculateBatchedParameterSize() override {
    batched_param_sizes_[0] = batch_size_ * sizeof(uint8*); // in_ptrs
    batched_param_sizes_[1] = batch_size_ * sizeof(int); // in_strides
    batched_param_sizes_[2] = batch_size_ * sizeof(bool); // mirror params
  }

  // Even though our parameter setup is embarrassingly parallel, we do it
  // in the serial method so that we can validate the parameters w/ the
  // ndll provided function
  inline void SerialBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend>* output) override {
    // Copy in_strides
    std::memcpy(batched_param_buffers_[1].template data<int>(),
        in_strides_.data(), in_strides_.size()*sizeof(int));

    // Calculate input ptrs & copy mirror flags
    for (int i = 0; i < batch_size_; ++i) {
      batched_param_buffers_[2].template data<bool>()[i] = mirror_[i];
      batched_param_buffers_[0].template data<const uint8*>()[i] =
        input.template datum<uint8>(i) + crop_offsets_[i];
    }

    // Validate parameters
    NDLL_CALL(ValidateBatchedCropMirrorNormalizePermute(
            batched_param_buffers_[0].template data<const uint8*>(),
            batched_param_buffers_[1].template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            batched_param_buffers_[2].template data<bool>(),
            mean_vec_.data(), std_vec_.data(),
            output->template data<OUT>()));
  }
  
  std::mt19937 rand_gen_;
  
  // Crop meta-data
  bool random_crop_;
  int crop_h_, crop_w_;

  // Mirror meta-data
  float mirror_prob_;

  // Input/output channel meta-data
  bool color_;
  int C_;

  // Staging area for kernel parameters
  vector<int> crop_offsets_;
  vector<int> in_strides_;
  vector<bool> mirror_;

  // Tensor to store mean & stddiv 
  Tensor<Backend> mean_, std_;
  vector<float> mean_vec_, std_vec_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
  using Operator<Backend>::batched_param_sizes_;
  using Operator<Backend>::batched_param_buffers_;
  using Operator<Backend>::batched_param_gpu_buffers_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_
