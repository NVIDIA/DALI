#ifndef NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_
#define NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_

#include <cstring>

#include <random>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/transformer.h"

namespace ndll {

template <typename Backend>
class CropMirrorNormalizePermuteOp : public Transformer<Backend> {
public:
  inline CropMirrorNormalizePermuteOp(const OpSpec &spec) :
    Transformer<Backend>(spec), rand_gen_(time(nullptr)),
    output_type_(spec.GetSingleArgument<NDLLDataType>("output_type", NDLL_FLOAT)),
    random_crop_(spec.GetSingleArgument<bool>("random_crop", false)),
    crop_h_(spec.GetSingleArgument<int>("crop_h", -1)),
    crop_w_(spec.GetSingleArgument<int>("crop_w", -1)),
    mirror_prob_(spec.GetSingleArgument<float>("mirror_prob", 0.5f)),
    image_type_(spec.GetSingleArgument<NDLLImageType>("image_type", NDLL_RGB)),
    color_(IsColor(image_type_)),
    C_(color_ ? 3 : 1),
    mean_vec_(spec.GetRepeatedArgument<float>("mean")),
    inv_std_vec_(spec.GetRepeatedArgument<float>("std")) {
    // Validate input parameters
    NDLL_ENFORCE(crop_h_ > 0 && crop_w_ > 0);
    NDLL_ENFORCE(mirror_prob_ <= 1.f && mirror_prob_ >= 0.f);

    // Validate & save mean & std params
    NDLL_ENFORCE((int)mean_vec_.size() == C_);
    NDLL_ENFORCE((int)inv_std_vec_.size() == C_);
    
    // Inverse the std-deviation
    for (int i = 0; i < C_; ++i) {
      inv_std_vec_[i] = 1.f / inv_std_vec_[i];
    }
    mean_.Copy(mean_vec_, stream_);
    inv_std_.Copy(inv_std_vec_, stream_);

    // We need three buffers for our batched parameters
    // in_ptrs, in_strides, and mirror parameters per image
    param_sizes_.resize(3);

    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    in_strides_.resize(batch_size_);
    mirror_.resize(batch_size_);
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
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    if (output_type_ == NDLL_FLOAT) {
      output->template mutable_data<float>();
    } else if (output_type_ == NDLL_FLOAT16) {
      output->template mutable_data<float16>();
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }
  
  inline string name() const override {
    return "CropMirrorNormalizePermuteOp";
  }

protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    if (output_type_ == NDLL_FLOAT) {
      RunHelper<float>(output);
    } else if (output_type_ == NDLL_FLOAT16) {
      RunHelper<float16>(output);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  template <typename OUT>
  inline void RunHelper(Batch<Backend> *output) {
    NDLL_CALL(BatchedCropMirrorNormalizePermute(
            gpu_param_buffers_[0].template data<const uint8*>(),
            gpu_param_buffers_[1].template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            gpu_param_buffers_[2].template data<bool>(),
            mean_.template data<float>(),
            inv_std_.template data<float>(),
            output->template mutable_data<OUT>(),
            stream_));
  }
  
  inline void CalculateBatchedParameterSize() override {
    param_sizes_[0] = batch_size_ * sizeof(uint8*); // in_ptrs
    param_sizes_[1] = batch_size_ * sizeof(int); // in_strides
    param_sizes_[2] = batch_size_ * sizeof(bool); // mirror params
  }

  // Even though our parameter setup is embarrassingly parallel, we do it
  // in the serial method so that we can validate the parameters w/ the
  // ndll provided function
  inline void SerialBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend>* output) override {
    // Copy in_strides
    std::memcpy(param_buffers_[1].template mutable_data<int>(),
        in_strides_.data(), in_strides_.size()*sizeof(int));

    // Calculate input ptrs & copy mirror flags
    for (int i = 0; i < batch_size_; ++i) {
      param_buffers_[2].template mutable_data<bool>()[i] = mirror_[i];
      param_buffers_[0].template mutable_data<const uint8*>()[i] =
        input.template sample<uint8>(i) + crop_offsets_[i];
    }

    if (output_type_ == NDLL_FLOAT) {
      ValidateHelper<float>(output);
    } else if (output_type_ == NDLL_FLOAT16) {
      ValidateHelper<float16>(output);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  template <typename OUT>
  inline void ValidateHelper(Batch<Backend> *output) {
    // Validate parameters
    NDLL_CALL(ValidateBatchedCropMirrorNormalizePermute(
            param_buffers_[0].template mutable_data<const uint8*>(),
            param_buffers_[1].template mutable_data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            param_buffers_[2].template mutable_data<bool>(),
            mean_vec_.data(), inv_std_vec_.data(),
            output->template mutable_data<OUT>()));
  }
  
  std::mt19937 rand_gen_;

  // Output data type
  NDLLDataType output_type_;
  
  // Crop meta-data
  bool random_crop_;
  int crop_h_, crop_w_;

  // Mirror meta-data
  float mirror_prob_;

  // Input/output channel meta-data
  NDLLImageType image_type_;
  bool color_;
  int C_;

  // Staging area for kernel parameters
  vector<int> crop_offsets_;
  vector<int> in_strides_;
  vector<bool> mirror_;

  // Tensor to store mean & stddiv 
  Tensor<Backend> mean_, inv_std_;
  vector<float> mean_vec_, inv_std_vec_;

  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_;
  using Operator<Backend>::param_sizes_;
  using Operator<Backend>::param_buffers_;
  using Operator<Backend>::gpu_param_buffers_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_PERMUTE_OP_H_
