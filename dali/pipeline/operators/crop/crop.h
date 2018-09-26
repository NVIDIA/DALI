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
  explicit inline CropAttr(const OpSpec &spec) :
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    C_(IsColor(image_type_) ? 3 : 1) {
      if (spec.name() != "Resize") {
        vector<int>cropTmp;
        GetSingleOrRepeatedArg(spec, &cropTmp, "crop", 2);
        crop_[0] = cropTmp[0];
        crop_[1] = cropTmp[1];
        DALI_ENFORCE(crop_[0] > 0 && crop_[1] > 0);
      }
    }

  std::pair<int, int> SetCropXY(const OpSpec &spec, const ArgumentWorkspace *ws,
     const Index imgIdx, int H, int W) const {
    DALI_ENFORCE(H >= crop_[0]);
    DALI_ENFORCE(W >= crop_[1]);

    const float crop_x_normalized = spec.GetArgument<float>("crop_pos_x", ws, imgIdx);
    const float crop_y_normalized = spec.GetArgument<float>("crop_pos_y", ws, imgIdx);

    DALI_ENFORCE(crop_y_normalized >= 0.f &&  crop_y_normalized <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");
    DALI_ENFORCE(crop_x_normalized >= 0.f &&  crop_x_normalized <= 1.f,
                 "Crop coordinates need to be in range [0.0, 1.0]");

    const int crop_y = crop_y_normalized * (H - crop_[0]);
    const int crop_x = crop_x_normalized * (W - crop_[1]);
    return std::make_pair(crop_y, crop_x);
  }

  const vector<Index> CheckShapes(const SampleWorkspace *ws) {
    const auto &input = ws->Input<CPUBackend>(0);

    // enforce that all shapes match
    for (int i = 1; i < ws->NumInput(); ++i) {
      DALI_ENFORCE(input.SameShape(ws->Input<CPUBackend>(i)));
    }

    DALI_ENFORCE(input.shape().size() == 3,
                 "Expects 3-dimensional image input.");

    return input.shape();
  }

  // Crop meta-data
  array<int, 2>crop_ = {{0}};

  const DALIImageType image_type_;
  const int C_;
};

template <typename Backend>
class Crop : public Operator<Backend>, protected CropAttr {
 public:
  explicit inline Crop(const OpSpec &spec) : Operator<Backend>(spec), CropAttr(spec) {
    // Resize per-image data
    crop_offsets_.resize(batch_size_);
    input_ptrs_.Resize({batch_size_});
    input_strides_.Resize({batch_size_});
    Init(batch_size_);
  }

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  virtual int GetPad() const                      { return C_; }

  inline const uint8 * const *InputImgsBatch() const {
    return input_ptrs_gpu_.template data<const uint8*>();
  }

  inline const int *InputStridesBatch() const {
    return input_strides_gpu_.template data<int>();
  }

  template <typename Out>
  void ValidateHelper(TensorList<Backend> *output);

  template <typename Out>
  Tensor<Backend> *PrepareCropParam(Workspace<Backend> *ws, const int idx,
                 const unsigned char **input_ptr, int *pStride, Out **pOutput_ptr) const;

 private:
  template <typename Out>
  void RunHelper(Workspace<Backend> *ws, const int idx);

  inline Dims GetOutShape(DALITensorLayout inputLayout, DALITensorLayout *pOutLayout) {
    *pOutLayout = output_layout_ == DALI_SAME ? inputLayout : output_layout_;
    const int pad_C = GetPad();
    if (*pOutLayout == DALI_NCHW)
      return {pad_C, crop_[0], crop_[1]};
    else
      return {crop_[0], crop_[1], pad_C};
  }

  void SetupSharedSampleParams(const ArgumentWorkspace *ws, const vector<Index> &inputShape,
                                int threaIdx, int dataIdx) {
    DALI_ENFORCE(inputShape.size() == 3, "Expects 3-dimensional image input.");

    const int H = inputShape[0];
    const int W = inputShape[1];

    per_sample_dimensions_[threaIdx] = std::make_pair(H, W);

    const int C = inputShape[2];
    DALI_ENFORCE(C == C_,
                 "Input channel dimension does not match "
                 "the output image type. Expected input with "
                 + to_string(C_) + " channels, got " + to_string(C) + ".");

    per_sample_crop_[threaIdx] = SetCropXY(spec_, ws, dataIdx, H, W);
  }

  void Init(int size) {
    per_sample_crop_.resize(size);
    per_sample_dimensions_.resize(size);
    output_type_ = DALI_NO_TYPE;
    output_layout_ = DALI_SAME;
  }

  void CallRunHelper(Workspace<Backend> *ws, const int idx) {
    DALI_IMAGE_TYPE_SWITCH_NO_FLOAT16(output_type_, imgType,
                                      RunHelper<imgType>(ws, idx));
  }

  Tensor<CPUBackend> input_ptrs_, input_strides_;
  Tensor<GPUBackend> input_ptrs_gpu_, input_strides_gpu_;
  vector<int> crop_offsets_;

  std::vector<std::pair<int, int>> per_sample_crop_;
  std::vector<std::pair<int, int>> per_sample_dimensions_;

 protected:
  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_CROP_H_

