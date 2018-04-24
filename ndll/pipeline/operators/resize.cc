// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

// NDLL_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(Resize, Resize<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA(Resize)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
  .AddOptionalArg("warp_resize", "Foo", false)
  .AddArg("resize_a", "Lower bound for resize")
  .AddArg("resize_b", "Upper bound for resize")
  .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
  .AddOptionalArg("crop", "Size of the cropped image", -1)
  .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB)
  .AddOptionalArg("interp_type", "Type of interpolation used", NDLL_INTERP_LINEAR);


void ResizeAttr::SetSize(NDLLSize *in_size, const vector<Index> &shape,
                         const resize_t &rand, NDLLSize *out_size) const {
    in_size->height = shape[0];
    in_size->width = shape[1];

    const resize_t &resize = random_resize_? rand : resize_;
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

void ResizeAttr::DefineCrop(NDLLSize *out_size, int *pCropX, int *pCropY) const {
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

}  // namespace ndll
