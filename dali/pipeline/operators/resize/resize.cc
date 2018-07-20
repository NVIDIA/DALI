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

#include "dali/pipeline/operators/resize/resize.h"

namespace dali {

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("resize_x", "The length of the X dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_y` is left at 0, then the op will keep "
      "the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_y", "The length of the Y dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`. "
      "If the `resize_x` is left at 0, then the op will keep "
      "the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_shorter", "The length of the shorter dimension of the resized image. "
      "This option is mutually exclusive with `resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image.", 0.f, true);

DALI_SCHEMA(Resize)
  .DocStr(R"code(Resize images.)code")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("save_attrs"));
  })
  .AllowMultipleInputSets()
  .AddOptionalArg("save_attrs",
      R"code(Save reshape attributes for testing.)code", false)
  .AddParent("ResizeAttr");

void ResizeAttr::SetSize(DALISize *in_size, const vector<Index> &shape, int idx,
                         DALISize *out_size, TransformMeta const *meta) const {
  in_size->height = shape[0];
  in_size->width = shape[1];

  if (!meta)
    meta = per_sample_meta_.data();

  out_size->height = meta[idx].rsz_h;
  out_size->width = meta[idx].rsz_w;
}

void ResizeAttr::DefineCrop(DALISize *out_size, int *pCropX, int *pCropY, int idx) const {
  *pCropX = per_sample_meta_[idx].crop_x;
  *pCropY = per_sample_meta_[idx].crop_y;
  out_size->height = crop_[0];
  out_size->width  = crop_[1];
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
