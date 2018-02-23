// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/resize.h"

namespace ndll {

NDLL_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);
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
  .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB)
  .AddOptionalArg("interp_type", "Type of interpolation used", NDLL_INTERP_LINEAR);


void ResizeAttr::SetSize(NDLLSize &in_size, const vector<Index> &shape,
                         const resize_t &rand, NDLLSize &out_size) {
    in_size.height = shape[0];
    in_size.width = shape[1];

    out_size.height = rand.first;
    if (warp_resize_) {
        out_size.width = rand.second;
        return;
    }

    if (in_size.width < in_size.height) {
        out_size.width = out_size.height;
        out_size.height =
                static_cast<float>(in_size.height) / in_size.width * out_size.width;
    } else {
        out_size.width =
                static_cast<float>(in_size.width) / in_size.height * out_size.height;
    }
}

}  // namespace ndll
