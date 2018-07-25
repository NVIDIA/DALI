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


#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_CAST_PERMUTE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_CAST_PERMUTE_H_

#include <vector>
#include <utility>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

template <typename Backend>
class CropCastPermute : public Operator<Backend> {
 public:
  explicit inline CropCastPermute(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<DALIDataType>("output_dtype")),
    output_layout_(spec.GetArgument<DALITensorLayout>("output_layout")),
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    color_(IsColor(image_type_)),
    C_(color_ ? 3 : 1) {
    vector<int> temp_crop;
    GetSingleOrRepeatedArg(spec, &temp_crop, "crop");

    crop_h_ = temp_crop[0];
    crop_w_ = temp_crop[1];

    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});

    per_sample_crop_.resize(batch_size_);
    per_sample_dimensions_.resize(batch_size_);
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  template <typename Out>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  template <typename Out>
  void ValidateHelper(TensorList<Backend> *output);

  int crop_h_;
  int crop_w_;

  // Input/output channel meta-data
  DALIImageType image_type_;
  bool color_;
  int C_;

  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;

  Tensor<CPUBackend> input_ptrs_, input_strides_;
  Tensor<GPUBackend> input_ptrs_gpu_, input_strides_gpu_;
  vector<int> crop_offsets_;

  // store per-thread crop offsets for same resize on multiple data
  std::vector<std::pair<int, int>> per_sample_crop_;
  std::vector<std::pair<int, int>> per_sample_dimensions_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_CAST_PERMUTE_H_

