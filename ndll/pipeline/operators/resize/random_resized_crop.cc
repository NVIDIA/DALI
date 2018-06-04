// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <vector>
#include <random>

#include "ndll/pipeline/operators/resize/random_resized_crop.h"
#include "ndll/util/ocv.h"

namespace ndll {

NDLL_SCHEMA(RandomResizedCrop)
  .DocStr("Perform a crop with randomly chosen area and aspect ratio, then resize it to given size.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("random_aspect_ratio", "Range from which to choose random aspect ratio", std::vector<float>{3./4., 4./3.})
  .AddOptionalArg("random_area", "Range from which to choose random area factor `A`."
      " Before resizing, the cropped image's area will be equal to `A` * original image's area.",
      std::vector<float>{0.08, 1.0})
  .AddOptionalArg("interp_type", "Interpolation method", NDLL_INTERP_LINEAR)
  .AddArg("size", "Size of resized image")
  .AddOptionalArg("num_attempts",
      "Maximum number of attempts used to choose random area and aspect ratio", 10);

template<>
struct RandomResizedCrop<CPUBackend>::Params {
  std::vector<std::mt19937> rand_gens;
  std::vector<std::uniform_real_distribution<float>> aspect_ratio_dis;
  std::vector<std::uniform_real_distribution<float>> area_dis;
  std::vector<std::uniform_real_distribution<float>> uniform;

  std::vector<CropInfo> crops;
};

template<>
void RandomResizedCrop<CPUBackend>::InitParams(const OpSpec &spec) {
  params_->rand_gens.resize(batch_size_);
  std::seed_seq seq{spec.GetArgument<int>("seed")};
  std::vector<int> seeds(batch_size_);
  seq.generate(seeds.begin(), seeds.end());
  for (size_t i = 0; i < seeds.size(); ++i) {
    params_->rand_gens[i].seed(seeds[i]);
  }
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
  params_->aspect_ratio_dis.resize(batch_size_);
  params_->area_dis.resize(batch_size_);
  for (size_t i = 0; i < params_->aspect_ratio_dis.size(); ++i) {
    params_->aspect_ratio_dis[i] = std::uniform_real_distribution<float>(aspect_ratios[0],
                                                                         aspect_ratios[1]);
    params_->area_dis[i] = std::uniform_real_distribution<float>(area[0],
                                                                 area[1]);
    params_->uniform[i] = std::uniform_real_distribution<float>(0, 1);
  }

  params_->crops.resize(batch_size_);
}

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(SampleWorkspace * ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  NDLL_ENFORCE(IsType<uint8>(input.type()),
      "Expected input data as uint8.");

  const int W = input.shape()[1];
  const int C = input.shape()[2];

  const int newH = size_[0];
  const int newW = size_[1];

  auto *output = ws->Output<CPUBackend>(idx);

  output->set_type(input.type());
  output->Resize({newH, newW, C});

  const CropInfo &crop = params_->crops[ws->data_idx()];
  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const uint8_t *img = input.data<uint8_t>();

  // Crop
  const cv::Mat cv_input_roi = CreateMatFromPtr(crop.h, crop.w,
                                                channel_flag,
                                                img + crop.y*W*C + crop.x*C,
                                                W*C);

  cv::Mat cv_output = CreateMatFromPtr(newH, newW,
                                       channel_flag,
                                       output->mutable_data<uint8_t>());

  int ocv_interp_type;
  NDLL_ENFORCE(OCVInterpForNDLLInterp(interp_type_, &ocv_interp_type) == NDLLSuccess,
      "Unknown interpolation type");
  cv::resize(cv_input_roi, cv_output, cv::Size(newW, newH), 0, 0, ocv_interp_type);
}

template<>
void RandomResizedCrop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  auto &input = ws->Input<CPUBackend>(0);
  vector<Index> input_shape = input.shape();
  NDLL_ENFORCE(input_shape.size() == 3,
      "Expects 3-dimensional image input.");

  int H = input_shape[0];
  int W = input_shape[1];

  CropInfo crop;
  int attempt = 0;
  int id = ws->data_idx();

  for (attempt = 0; attempt < num_attempts_; ++attempt) {
    if (TryCrop(H, W,
                &params_->aspect_ratio_dis[id],
                &params_->area_dis[id],
                &params_->uniform[id],
                &params_->rand_gens[id],
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

  params_->crops[id] = crop;
}

NDLL_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace ndll
