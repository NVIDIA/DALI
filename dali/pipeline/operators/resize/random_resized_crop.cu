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
#include <cmath>

#include "dali/util/npp.h"
#include "dali/pipeline/operators/resize/random_resized_crop.h"

namespace dali {

template<>
struct RandomResizedCrop<GPUBackend>::Params {
  std::mt19937 rand_gen;
  std::uniform_real_distribution<float> aspect_ratio_dis;
  std::uniform_real_distribution<float> area_dis;
  std::uniform_real_distribution<float> uniform;

  std::vector<CropInfo> crops;
};

template<>
void RandomResizedCrop<GPUBackend>::InitParams(const OpSpec &spec) {
  params_->rand_gen.seed(spec.GetArgument<int64_t>("seed"));
  params_->aspect_ratio_dis = std::uniform_real_distribution<float>(aspect_ratios_[0],
                                                                    aspect_ratios_[1]);
  params_->area_dis = std::uniform_real_distribution<float>(area_[0],
                                                            area_[1]);
  params_->uniform = std::uniform_real_distribution<float>(0, 1);

  params_->crops.resize(batch_size_);
}

template<>
void RandomResizedCrop<GPUBackend>::RunImpl(DeviceWorkspace * ws, const int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  const int newH = size_[0];
  const int newW = size_[1];

  auto *output = ws->Output<GPUBackend>(idx);
  output->set_type(input.type());

  std::vector<Dims> output_shape(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    const int C = input.tensor_shape(i)[2];
    output_shape[i] = {newH, newW, C};
  }
  output->Resize(output_shape);

  cudaStream_t old_stream = nppGetStream();
  nppSetStream(ws->stream());

  for (int i = 0; i < batch_size_; ++i) {
    const CropInfo &crop = params_->crops[i];
    NppiRect in_roi, out_roi;
    in_roi.x = crop.x;
    in_roi.y = crop.y;
    in_roi.width = crop.w;
    in_roi.height = crop.h;
    out_roi.x = 0;
    out_roi.y = 0;
    out_roi.width = newW;
    out_roi.height = newH;

    const int H = input.tensor_shape(i)[0];  // HWC
    const int W = input.tensor_shape(i)[1];  // HWC
    const int C = input.tensor_shape(i)[2];  // HWC

    NppiSize input_size, output_size;

    input_size.width = W;
    input_size.height = H;

    output_size.width = newW;
    output_size.height = newH;

    NppiInterpolationMode npp_interp_type;
    DALI_ENFORCE(NPPInterpForDALIInterp(interp_type_, &npp_interp_type) == DALISuccess,
        "Unsupported interpolation type");

    switch (C) {
      case 3:
        DALI_CHECK_NPP(nppiResize_8u_C3R(input.tensor<uint8_t>(i),
                                         W*C,
                                         input_size,
                                         in_roi,
                                         output->mutable_tensor<uint8_t>(i),
                                         newW*C,
                                         output_size,
                                         out_roi,
                                         npp_interp_type));
        break;
      case 1:
        DALI_CHECK_NPP(nppiResize_8u_C1R(input.tensor<uint8_t>(i),
                                         W*C,
                                         input_size,
                                         in_roi,
                                         output->mutable_tensor<uint8_t>(i),
                                         newW*C,
                                         output_size,
                                         out_roi,
                                         npp_interp_type));
        break;
      default:
        DALI_FAIL("RandomResizedCrop is implemented only for images"
            " with C = 1 or 3, but encountered C = " + to_string(C) + ".");
    }
  }
  nppSetStream(old_stream);
}

template<>
void RandomResizedCrop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");
  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3,
        "Expects 3-dimensional image input.");

    int H = input_shape[0];
    int W = input_shape[1];

    CropInfo crop;
    int attempt = 0;

    for (attempt = 0; attempt < num_attempts_; ++attempt) {
      if (TryCrop(H, W,
                  &params_->aspect_ratio_dis,
                  &params_->area_dis,
                  &params_->uniform,
                  &params_->rand_gen,
                  &crop)) {
        break;
      }
    }

    if (attempt == num_attempts_) {
      int min_dim = H < W ? H : W;
      crop.w = min_dim;
      crop.h = min_dim;
      crop.x = (W - min_dim) / 2;
      crop.y = (H - min_dim) / 2;
    }

    params_->crops[i] = crop;
  }
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<GPUBackend>, GPU);

}  // namespace dali
