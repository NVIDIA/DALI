// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/resize/resize.h"

namespace dali {

DALI_SCHEMA(Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .AllowMultipleInputSets()
  .AddOptionalArg("random_resize",
      R"code(`bool`
      Whether to randomly resize images.)code", false)
  .AddOptionalArg("warp_resize",
      R"code(`bool`
      Whether to modify the aspect ratio of the image.)code", false)
  .AddArg("resize_a",
      R"code(`int`
      If neither `random_resize` nor `warp_resize` is set - size to which the shorter side of the image is resized.
      If `warp_image` is set and `random_resize` is not set - size to which height of the image is resized.
      If `random_resize` is set and `warp_resize` is not set - lower bound for the shorter side of the resized image.
      If both `random_resize` and `warp_resize` are set - lower bound for resized image's height and width.)code")
  .AddArg("resize_b",
      R"code(`int`
      If neither `random_resize` nor `warp_resize` is set - ignored.
      If `warp_image` is set and `random_resize` is not set - size to which width of the image is resized.
      If `random_resize` is set and `warp_resize` is not set - upper bound for the shorter side of the resized image.
      If both `random_resize` and `warp_resize` are set - upper bound for resized image's height and width.)code")
//  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
//  .AddOptionalArg("crop", "Size of the cropped image", -1)
//  .AddOptionalArg("mirror_prob", "Probability of a random horizontal or "
//      "vertical flip of the image", vector<float>{0.f, 0.f})
  .AddOptionalArg("image_type",
        R"code(`dali.types.DALIImageType`
        The color space of input and output image)code", DALI_RGB)
  .AddOptionalArg("interp_type",
      R"code(`dali.types.DALIInterpType`
      Type of interpolation used)code",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("save_attrs",
      R"code(`bool`
      Save reshape attributes for testing)code", false);


resize_t ResizeAttr::GetRandomSizes() const {
  if (!random_resize_)
      return resize_;

  auto rand_a = std::uniform_int_distribution<>(resize_.first, resize_.second)(rand_gen_);
  auto rand_b = std::uniform_int_distribution<>(resize_.first, resize_.second)(rand_gen_);
  return std::make_pair(rand_a, rand_b);
}

void ResizeAttr::SetSize(DALISize *in_size, const vector<Index> &shape,
                         const resize_t &rand, DALISize *out_size) const {
  in_size->height = shape[0];
  in_size->width = shape[1];

  const resize_t &resize = random_resize_ ? rand : resize_;
  if (warp_resize_) {
    out_size->height = resize.first;
    out_size->width = resize.second;
    return;
  }

  const float prop = static_cast<float>(in_size->height) / in_size->width;
  if (prop > 1.) {
    out_size->width = resize.first;
    out_size->height = prop * out_size->width;
  } else {
    out_size->height = resize.first;
    out_size->width = out_size->height / prop;
  }
}

void ResizeAttr::DefineCrop(DALISize *out_size, int *pCropX, int *pCropY) const {
  // Set crop parameters
  if (random_crop_) {
    *pCropX = randomUniform(out_size->width - crop_[0]);
    *pCropY = randomUniform(out_size->height - crop_[1]);
  } else {
    *pCropX = (out_size->width - crop_[0]) / 2;
    *pCropY = (out_size->height - crop_[1]) / 2;
  }

  out_size->width = crop_[0];
  out_size->height = crop_[1];
}

template <>
void Resize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *) {
  DALI_FAIL("Not implemented");
}

template <>
void Resize<CPUBackend>::RunImpl(SampleWorkspace *, const int) {
  DALI_FAIL("Not implemented");
}

}  // namespace dali
