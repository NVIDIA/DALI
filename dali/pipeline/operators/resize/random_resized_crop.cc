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


#include <vector>
#include <random>

#include "dali/pipeline/operators/resize/random_resized_crop.h"
#include "dali/pipeline/operators/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(RandomResizedCrop)
  .DocStr("Perform a crop with randomly chosen area and aspect ratio,"
      " then resize it to given size.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("random_aspect_ratio",
      R"code(Range from which to choose random aspect ratio (width/height).)code",
      std::vector<float>{3./4., 4./3.})
  .AddOptionalArg("random_area",
      R"code(Range from which to choose random area factor `A`.
Before resizing, the cropped image's area will be equal to `A` * original image's area.)code",
      std::vector<float>{0.08, 1.0})
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR)
  .AddArg("size",
      R"code(Size of resized image.)code",
      DALI_INT_VEC)
  .AddOptionalArg("num_attempts",
      R"code(Maximum number of attempts used to choose random area and aspect ratio.)code",
      10)
  .EnforceInputLayout(DALI_NHWC);

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(SampleWorkspace * ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  const int W = input.shape()[1];
  const int C = input.shape()[2];

  const int newH = size_[0];
  const int newW = size_[1];

  auto &output = ws->Output<CPUBackend>(idx);

  output.set_type(input.type());
  output.Resize({newH, newW, C});

  const CropWindow &crop = params_->crops[ws->data_idx()];
  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const uint8_t *img = input.data<uint8_t>();

  // Crop
  const cv::Mat cv_input_roi = CreateMatFromPtr(crop.h, crop.w,
                                                channel_flag,
                                                img + crop.y*W*C + crop.x*C,
                                                W*C);

  cv::Mat cv_output = CreateMatFromPtr(newH, newW,
                                       channel_flag,
                                       output.mutable_data<uint8_t>());

  int ocv_interp_type;
  DALI_ENFORCE(OCVInterpForDALIInterp(interp_type_, &ocv_interp_type) == DALISuccess,
      "Unknown interpolation type");
  cv::resize(cv_input_roi, cv_output, cv::Size(newW, newH), 0, 0, ocv_interp_type);
}

template<>
void RandomResizedCrop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  auto &input = ws->Input<CPUBackend>(0);
  vector<Index> input_shape = input.shape();
  DALI_ENFORCE(input_shape.size() == 3,
      "Expects 3-dimensional image input.");

  int H = input_shape[0];
  int W = input_shape[1];
  int id = ws->data_idx();

  params_->crops[id] = params_->crop_gens[id].GenerateCropWindow(H, W);
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace dali
