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


#include "dali/pipeline/operators/resize/new_resize.h"

namespace dali {

template <>
void NewResize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto &output = ws->Output<CPUBackend>(idx);

  const auto &input_shape = input.shape();
  DALISize out_size, input_size;
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
void NewResize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *) {}

// DALI_REGISTER_OPERATOR(NewResize, NewResize<CPUBackend>, CPU);
// DALI_REGISTER_OPERATOR(NewResize, NewResize<GPUBackend>, GPU);
// DALI_SCHEMA(NewResize)
    // .DocStr("Resize images. Can do both fixed and random resizes, along with fused"
            // "cropping (random and fixed) and image mirroring.")
    // .NumInput(1)
    // .NumOutput(1)
    // .AddOptionalArg("random_resize", "Whether to randomly resize images", false)
    // .AddOptionalArg("warp_resize", "Foo", false)
    // .AddArg("resize_a", "Lower bound for resize")
    // .AddArg("resize_b", "Upper bound for resize")
    // .AddOptionalArg("image_type", "Type of the input image", DALI_RGB)
    // .AddOptionalArg("random_crop", "Whether to randomly choose the position of the crop", false)
    // .AddOptionalArg("crop", "Size of the cropped image", -1)
    // .AddOptionalArg("mirror_prob", "Probability of a random horizontal or "
                    // "vertical flip of the image", vector<float>{0.f, 0.f})
    // .AddOptionalArg("interp_type", "Type of interpolation used", DALI_INTERP_LINEAR);

}  // namespace dali

