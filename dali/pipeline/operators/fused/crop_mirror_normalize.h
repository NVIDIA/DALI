// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/crop/crop_attr.h"

namespace dali {

template <typename Backend>
class CropMirrorNormalize : public Operator<Backend>, protected CropAttr  {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec) :
    Operator<Backend>(spec),
    CropAttr(spec),
    output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
    output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
    pad_(spec.GetArgument<bool>("pad_output")),
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    color_(IsColor(image_type_)),
    C_(color_ ? 3 : 1) {
    vector<float> temp_crop;
    GetSingleOrRepeatedArg(spec, temp_crop, "crop", 2);

    crop_h_ = temp_crop[0];
    crop_w_ = temp_crop[1];

    has_mirror_ = spec.HasTensorArgument("mirror");
    if (!has_mirror_) {
      mirror_.Resize({batch_size_});
      for (int i = 0; i < batch_size_; ++i) {
        mirror_.mutable_data<int>()[i] = spec.GetArgument<int>("mirror");
      }
    }

    DALI_ENFORCE(crop_h_ > 0 && crop_w_ > 0);

    GetSingleOrRepeatedArg(spec, mean_vec_, "mean", C_);
    GetSingleOrRepeatedArg(spec, inv_std_vec_, "std", C_);

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

  inline ~CropMirrorNormalize() override = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  template <typename OUT>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  template <typename OUT>
  void ValidateHelper(TensorList<Backend> &output);

  inline Dims GetOutShape(DALITensorLayout inputLayout, DALITensorLayout *pOutLayout) {
    *pOutLayout = output_layout_ == DALI_SAME ? inputLayout : output_layout_;
    if (*pOutLayout == DALI_NCHW)
      return {C_, crop_h_, crop_w_};
    else
      return {crop_h_, crop_w_, C_};
  }

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
  using Operator<Backend>::RunImpl;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
