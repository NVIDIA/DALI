// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/resize/new_resize.h"

namespace ndll {

template <>
void NewResize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto &output = ws->Output<CPUBackend>(idx);

  const auto &input_shape = input.shape();
  NDLLSize out_size, input_size;
  SetSize(&input_size, input_shape, GetRandomSizes(), &out_size);

  const int C = input_shape[2];

  ResizeGridParam resizeParam[N_GRID_PARAMS] = {};
  ResizeMappingTable resizeTbl;
  PrepareCropAndResize(&input_size, &out_size, C, resizeParam, &resizeTbl);

  const int H0 = input_size.height;
  const int W0 = input_size.width;
  const int H1 = out_size.height;
  const int W1 = out_size.width;
  bool mirrorHor, mirrorVert;
  MirrorNeeded(&mirrorHor, &mirrorVert);

  DataDependentSetupCPU(input, output, "NewResize", NULL, NULL, NULL, &out_size);
  const auto pResizeMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingCPU);
  const auto pMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingSimpleCPU);
  const auto pPixMapping = PIX_MAPPING_CPU(resizeTbl.pixMappingCPU);
  AUGMENT_RESIZE_CPU(H1, W1, C, input.template data<uint8>(),
                   static_cast<uint8 *>(output->raw_mutable_data()), RESIZE_N);
}

template <>
void NewResize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {}

NDLL_REGISTER_OPERATOR(NewResize, NewResize<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(NewResize, NewResize<GPUBackend>, GPU);
NDLL_SCHEMA(NewResize)
    .DocStr("Resize images. Can do both fixed and random resizes, along with fused"
            "cropping (random and fixed) and image mirroring.")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
    .AddOptionalArg("warp_resize", "Foo", false)
    .AddArg("resize_a", "Lower bound for resize")
    .AddArg("resize_b", "Upper bound for resize")
    .AddOptionalArg("image_type", "Type of the input image", NDLL_RGB)
    .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
    .AddOptionalArg("crop", "Size of the cropped image", -1)
    .AddOptionalArg("mirror_prob", "Probability of a random horizontal or "
                    "vertical flip of the image", vector<float>{0.f, 0.f})
    .AddOptionalArg("interp_type", "Type of interpolation used", NDLL_INTERP_LINEAR);

NDLL_REGISTER_TYPE(ResizeMapping, NDLL_RESIZE_MAPPING);
NDLL_REGISTER_TYPE(PixMapping, NDLL_PIX_MAPPING);
NDLL_REGISTER_TYPE(uint32_t, NDLL_UINT32);
}  // namespace ndll

