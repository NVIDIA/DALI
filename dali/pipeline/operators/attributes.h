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


#ifndef DALI_PIPELINE_OPERATORS_ATTRIBUTES_H_
#define DALI_PIPELINE_OPERATORS_ATTRIBUTES_H_

#include <vector>
#include <utility>
#include "dali/pipeline/operators/common.h"
#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

class CastPermuteAttr {
 protected:
  explicit inline CastPermuteAttr(const OpSpec &spec, bool defaultCastPermut = true) :
    image_type_(spec.GetArgument<DALIImageType>("image_type")),
    C_(IsColor(image_type_) ? 3 : 1) {
    if (defaultCastPermut) {
      output_type_ = DALI_NO_TYPE;
      output_layout_ = DALI_SAME;
    } else {
      output_type_ = spec.GetArgument<DALIDataType>("output_dtype");
      output_layout_ = spec.GetArgument<DALITensorLayout>("output_layout");
    }
  }

  void SetupSharedSampleParams(const DeviceWorkspace *ws) {
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = ws->Input<GPUBackend>(0).type().id();
  }

  void SetupSharedSampleParams(const SampleWorkspace *ws) {
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = ws->Input<CPUBackend>(0).type().id();
  }

  // Output data type
  DALIDataType output_type_;

  // Output data layout
  DALITensorLayout output_layout_;
  const DALIImageType image_type_;
  const int C_;
};

class CropAttr : protected CastPermuteAttr {
 protected:
  explicit inline CropAttr(const OpSpec &spec, bool defaultCastPermut = true) :
                     CastPermuteAttr(spec, defaultCastPermut) {
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
};

template <typename Backend>
class NormalizeAttr {
 public:
  void InitNormalizeAttr(const OpSpec &spec, int C) {
    vector<float> mean_vec, inv_std_vec;
    GetSingleOrRepeatedArg(spec, &mean_vec, "mean", C);
    GetSingleOrRepeatedArg(spec, &inv_std_vec, "std", C);

    // Inverse the std-deviation
    for (int i = 0; i < C; ++i)
      inv_std_vec[i] = 1.f / inv_std_vec[i];

    mean_.Copy(mean_vec, 0);
    inv_std_.Copy(inv_std_vec, 0);
  }

 protected:
  // Tensor to store mean & stddiv
  Tensor<Backend> mean_, inv_std_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_ATTRIBUTES_H_

