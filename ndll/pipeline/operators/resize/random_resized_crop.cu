// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <vector>
#include <cmath>

#include "ndll/pipeline/operators/resize/random_resized_crop.h"

namespace ndll {

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
  params_->rand_gen.seed(spec.GetArgument<int>("seed"));
  std::vector<float> aspect_ratios = spec.GetRepeatedArgument<float>("random_aspect_ratio");
  std::vector<float> area = spec.GetRepeatedArgument<float>("random_area");
  NDLL_ENFORCE(aspect_ratios.size() == 2,
      "\"random_aspect_ratio\" argument should be a list of size 2");
  NDLL_ENFORCE(aspect_ratios[0] <= aspect_ratios[1],
      "Provided empty range");
  NDLL_ENFORCE(area.size() == 2,
      "\"random_area\" argument should be a list of size 2");
  NDLL_ENFORCE(area[0] <= area[1],
      "Provided empty range");
  params_->aspect_ratio_dis = std::uniform_real_distribution<float>(aspect_ratios[0],
                                                                    aspect_ratios[1]);
  params_->area_dis = std::uniform_real_distribution<float>(area[0],
                                                            area[1]);
  params_->uniform = std::uniform_real_distribution<float>(0, 1);

  params_->crops.resize(batch_size_);
}

template<>
void RandomResizedCrop<GPUBackend>::RunImpl(DeviceWorkspace * ws, const int idx) {
}

template<>
void RandomResizedCrop<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  NDLL_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");
  for (int i = 0; i < batch_size_; ++i) {
    vector<Index> input_shape = input.tensor_shape(i);
    NDLL_ENFORCE(input_shape.size() == 3,
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

NDLL_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<GPUBackend>, GPU);

}  // namespace ndll
