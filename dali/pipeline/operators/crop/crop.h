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

#ifndef DALI_PIPELINE_OPERATORS_CROP_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_CROP_H_

#include <vector>
#include <utility>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

class CropAttr {
 protected:
  explicit inline CropAttr(const OpSpec &spec)
      : image_type_(spec.GetArgument<DALIImageType>("image_type")),
        C_(IsColor(image_type_) ? 3 : 1),
        batch_size_{spec.GetArgument<int>("batch_size")} {
    if (spec.name() != "Resize") {
      vector<float> cropArgs = spec.GetRepeatedArgument<float>("crop");

      DALI_ENFORCE(cropArgs[0] >=0,
        "Crop height must be greater than zero. Received: " + std::to_string(cropArgs[0]));
      DALI_ENFORCE(cropArgs[1] >=0,
        "Crop width must be greater than zero. Received: " + std::to_string(cropArgs[0]));

      crop_height_ = std::vector<int>(batch_size_, static_cast<int>(cropArgs[0]));
      crop_width_ = std::vector<int>(batch_size_, static_cast<int>(cropArgs[1]));
    }
  }

  std::pair<int, int> SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
                                const Index dataIdx, int H, int W) {
    DALI_ENFORCE(H >= crop_height_[dataIdx]);
    DALI_ENFORCE(W >= crop_width_[dataIdx]);

    auto crop_x_norm =
        std::vector<float>(batch_size_, spec.GetArgument<float>("crop_pos_x", ws, dataIdx));
    auto crop_y_norm =
        std::vector<float>(batch_size_, spec.GetArgument<float>("crop_pos_y", ws, dataIdx));

    DALI_ENFORCE(crop_y_norm[dataIdx] >= 0.f && crop_y_norm[dataIdx] <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_norm[dataIdx] >= 0.f && crop_x_norm[dataIdx] <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");

    const int crop_y = crop_y_norm[dataIdx] * (H - crop_height_[dataIdx]);
    const int crop_x = crop_x_norm[dataIdx] * (W - crop_width_[dataIdx]);

    return std::make_pair(crop_y, crop_x);
  }

  const vector<Index> CheckShapes(const SampleWorkspace *ws) {
    const auto &input = ws->Input<CPUBackend>(0);

    DALI_ENFORCE(input.shape().size() == 3,
                 "Expects 3-dimensional image input.");

    return input.shape();
  }

  vector<int> crop_height_;
  vector<int> crop_width_;

  vector<float> crop_x_norm_;
  vector<float> crop_y_norm_;

  const DALIImageType image_type_;
  const int C_;
  const int batch_size_;
};

template <typename Backend>
class Crop : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline Crop(const OpSpec &spec)
      : Operator<Backend>(spec), CropAttr(spec) {
    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});
    Init(batch_size_);
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

 private:
  template <typename Out>
  void RunHelper(Workspace<Backend> *ws, int idx);
  void DataDependentSetup(Workspace<Backend> *ws, int idx);
  template <typename Out>
  void ValidateHelper(TensorList<Backend> *output, int idx);
  template <typename Out>
  void ValidateHelper(const Tensor<Backend> *input, Tensor<Backend> *output,
                      int idx);

  inline Dims GetOutShape(DALITensorLayout inputLayout,
                          DALITensorLayout *pOutLayout, int dataIdx) {
    *pOutLayout = output_layout_ == DALI_SAME ? inputLayout : output_layout_;
    if (*pOutLayout == DALI_NCHW)
      return {C_, crop_height_[dataIdx], crop_width_[dataIdx]};
    else
      return {crop_height_[dataIdx], crop_width_[dataIdx], C_};
  }

  void SetupSharedSampleParams(const ArgumentWorkspace *ws,
                               const vector<Index> &inputShape, int threadIdx,
                               int dataIdx) {
    DALI_ENFORCE(inputShape.size() == 3, "Expects 3-dimensional image input.");

    const int H = inputShape[0];
    const int W = inputShape[1];

    per_sample_dimensions_[threadIdx] = std::make_pair(H, W);

    int C = inputShape[2];
    DALI_ENFORCE(C == C_,
                 "Input channel dimension does not match "
                 "the output image type. Expected input with " +
                     to_string(C_) + " channels, got " + to_string(C) + ".");

    per_sample_crop_[threadIdx] = SetCropXY(spec_, ws, dataIdx, H, W);
  }

  void Init(int size) {
    per_sample_crop_.resize(size);
    per_sample_dimensions_.resize(size);
    output_type_ = DALI_NO_TYPE;
    output_layout_ = DALI_SAME;
  }

  void CallRunHelper(Workspace<Backend> *ws, int idx) {
    if (output_type_ == DALI_FLOAT) {
      RunHelper<float>(ws, idx);
    } else if (output_type_ == DALI_UINT8) {
      RunHelper<unsigned char>(ws, idx);
    } else if (output_type_ == DALI_INT16) {
      RunHelper<int16>(ws, idx);
    } else if (output_type_ == DALI_INT32) {
      RunHelper<int>(ws, idx);
    } else if (output_type_ == DALI_INT64) {
      RunHelper<int64>(ws, idx);
    } else {
      DALI_FAIL("Unsupported output type.");
    }
  }

  Tensor<CPUBackend> input_ptrs_, input_strides_;
  Tensor<GPUBackend> input_ptrs_gpu_, input_strides_gpu_;
  vector<int> crop_offsets_;

 protected:
  std::vector<std::pair<int, int>> per_sample_crop_;
  std::vector<std::pair<int, int>> per_sample_dimensions_;

  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_
