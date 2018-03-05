// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_H_
#define NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>
#include <random>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

NDLL_REGISTER_TYPE(const uint8*, NDLL_INTERNAL_C_UINT8_P);

template <typename Backend>
class CropMirrorNormalize : public Operator {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec) :
    Operator(spec), rand_gen_(time(nullptr)),
    output_type_(spec.GetArgument<NDLLDataType>("output_type", NDLL_FLOAT)),
    random_crop_(spec.GetArgument<bool>("random_crop", false)),
    crop_h_(spec.GetArgument<int>("crop_h", -1)),
    crop_w_(spec.GetArgument<int>("crop_w", -1)),
    mirror_prob_(spec.GetArgument<float>("mirror_prob", 0.5f)),
    image_type_(spec.GetArgument<NDLLImageType>("image_type", NDLL_RGB)),
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

    // TODO(tgale): We don't really want to do this in
    // the default stream, we should make at least some
    // stream available for constructors of ops.
    mean_.Copy(mean_vec_, 0);
    inv_std_.Copy(inv_std_vec_, 0);

    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});
    mirror_.Resize({batch_size_});

    // Reset per-set-of-samples random numbers
    per_sample_crop_.resize(batch_size_);
  }

  virtual inline ~CropMirrorNormalize() = default;

 protected:
  inline void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
    DataDependentSetup(ws, idx);
    if (output_type_ == NDLL_FLOAT) {
      RunHelper<float>(ws, idx);
    } else if (output_type_ == NDLL_FLOAT16) {
      RunHelper<float16>(ws, idx);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  inline void DataDependentSetup(DeviceWorkspace *ws, const int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto output = ws->Output<GPUBackend>(idx);
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Expected input data as uint8.");

    vector<Dims> output_shape(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
      vector<Index> input_shape = input.tensor_shape(i);
      NDLL_ENFORCE(input_shape.size() == 3,
          "Expects 3-dimensional image input.");
      NDLL_ENFORCE(input_shape[2] == C_,
          "Input channel dimension does not match "
          "the output channel argument.");

      int H = input_shape[0];
      int W = input_shape[1];

      NDLL_ENFORCE(H >= crop_h_);
      NDLL_ENFORCE(W >= crop_w_);

      // Set crop parameters
      int crop_y, crop_x;
      if (idx == 0) {
        // First set of samples determines the crop offsets to be used
        // for all sets of samples and stores.
        if (random_crop_) {
          crop_y = std::uniform_int_distribution<>(0, H - crop_h_)(rand_gen_);
          crop_x = std::uniform_int_distribution<>(0, W - crop_w_)(rand_gen_);
        } else {
          crop_y = (H - crop_h_) / 2;
          crop_x = (W - crop_w_) / 2;
        }
        per_sample_crop_[i] = std::make_pair(crop_y, crop_x);
      } else {
        // retrieve already determined offsets
        crop_y = per_sample_crop_[i].first;
        crop_x = per_sample_crop_[i].second;
      }

      // Save image stride & crop offset
      input_strides_.template mutable_data<int>()[i] = W*C_;
      crop_offsets_[i] = crop_y*W*C_ + crop_x*C_;

      // Set mirror parameters
      mirror_.template mutable_data<bool>()[i] =
        std::bernoulli_distribution(mirror_prob_)(rand_gen_);

      // Save the output shape of this image
      output_shape[i] = {C_, crop_h_, crop_w_};
    }

    // Resize the output data
    output->Resize(output_shape);

    // Copy strides and mirror data to gpu
    input_strides_gpu_.Copy(input_strides_, ws->stream());
    mirror_gpu_.Copy(mirror_, ws->stream());

    // Calculate input pointers and copy to gpu
    for (int i = 0; i < batch_size_; ++i) {
      input_ptrs_.template mutable_data<const uint8*>()[i] =
        input.template tensor<uint8>(i) + crop_offsets_[i];
    }
    input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());

    // Validate
    if (output_type_ == NDLL_FLOAT) {
      ValidateHelper<float>(output);
    } else if (output_type_ == NDLL_FLOAT16) {
      ValidateHelper<float16>(output);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  template <typename OUT>
  inline void RunHelper(DeviceWorkspace *ws, const int idx) {
    auto output = ws->Output<GPUBackend>(idx);
    NDLL_CALL(BatchedCropMirrorNormalizePermute(
            input_ptrs_gpu_.template data<const uint8*>(),
            input_strides_gpu_.template data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            mirror_gpu_.template data<bool>(),
            mean_.template data<float>(),
            inv_std_.template data<float>(),
            output->template mutable_data<OUT>(),
            ws->stream()));
  }

  template <typename OUT>
  inline void ValidateHelper(TensorList<Backend> *output) {
    // Validate parameters
    NDLL_CALL(ValidateBatchedCropMirrorNormalizePermute(
            input_ptrs_.template mutable_data<const uint8*>(),
            input_strides_.template mutable_data<int>(),
            batch_size_, crop_h_, crop_w_, C_,
            mirror_.template mutable_data<bool>(),
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

  Tensor<CPUBackend> input_ptrs_, input_strides_, mirror_;
  Tensor<GPUBackend> input_ptrs_gpu_, input_strides_gpu_, mirror_gpu_;
  vector<int> crop_offsets_;

  // Tensor to store mean & stddiv
  Tensor<Backend> mean_, inv_std_;
  vector<float> mean_vec_, inv_std_vec_;

  // store per-thread crop offsets for same resize on multiple data
  std::vector<std::pair<int, int>> per_sample_crop_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_CROP_MIRROR_NORMALIZE_H_
