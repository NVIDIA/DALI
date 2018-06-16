// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class CropMirrorNormalize : public Operator<Backend> {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
    output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
    pad_(spec.GetArgument<bool>("pad_output")),
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    color_(IsColor(image_type_)),
    C_(color_ ? 3 : 1),
    mean_vec_(spec.GetRepeatedArgument<float>("mean")),
    inv_std_vec_(spec.GetRepeatedArgument<float>("std")) {
    vector<int> temp_crop;
    GetSingleOrDoubleArg(spec, &temp_crop, "crop");

    DALI_ENFORCE(temp_crop.size() == 2, "Argument \"crop\" expects a list of at most 2 elements, "
        + to_string(temp_crop.size()) + " given.");
    crop_h_ = temp_crop[0];
    crop_w_ = temp_crop[1];

    has_mirror_ = spec.HasTensorArgument("mirror");
    if (!has_mirror_) {
      mirror_.Resize({batch_size_});
      for (int i = 0; i < batch_size_; ++i) {
        mirror_.mutable_data<int>()[i] = 0;
      }
    }

    // Validate input parameters
    DALI_ENFORCE(output_layout_ == DALI_NCHW ||
                 output_layout_ == DALI_NHWC,
                 "Unsupported output layout."
                 "Expected NCHW or NHWC.");
    DALI_ENFORCE(crop_h_ > 0 && crop_w_ > 0);

    // Validate & save mean & std params
    DALI_ENFORCE((int)mean_vec_.size() == C_);
    DALI_ENFORCE((int)inv_std_vec_.size() == C_);

    // Inverse the std-deviation
    for (int i = 0; i < C_; ++i) {
      inv_std_vec_[i] = 1.f / inv_std_vec_[i];
    }

    mean_.Copy(mean_vec_, 0);
    inv_std_.Copy(inv_std_vec_, 0);

    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});

    // Reset per-set-of-samples random numbers
    per_sample_crop_.resize(batch_size_);
    per_sample_dimensions_.resize(batch_size_);
  }

  virtual inline ~CropMirrorNormalize() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  template <typename OUT>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  template <typename OUT>
  void ValidateHelper(TensorList<Backend> *output);

  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;

  // Whether to pad output to 4 channels
  bool pad_;

  // Crop meta-data
  int crop_h_;
  int crop_w_;

  // Mirror?
  bool has_mirror_;

  // Input/output channel meta-data
  DALIImageType image_type_;
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
  std::vector<std::pair<int, int>> per_sample_dimensions_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
