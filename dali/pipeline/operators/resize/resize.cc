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
#include <opencv2/opencv.hpp>
#include "dali/util/ocv.h"

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
      "This option is mutually exclusive with `resize_longer`, `resize_x` and `resize_y`. "
      "The op will keep the aspect ratio of the original image.", 0.f, true)
  .AddOptionalArg("resize_longer", "The length of the longer dimension of the resized image. "
      "This option is mutually exclusive with `resize_shorter`,`resize_x` and `resize_y`. "
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
  *pCropX = per_sample_meta_[idx].crop.second;
  *pCropY = per_sample_meta_[idx].crop.first;
  out_size->height = crop_height_[idx];
  out_size->width  = crop_width_[idx];
}

template<>
Resize<CPUBackend>::Resize(const OpSpec &spec) : Operator<CPUBackend>(spec), ResizeAttr(spec) {
  per_sample_meta_.resize(num_threads_);
  save_attrs_ = spec_.HasArgument("save_attrs");
  outputs_per_idx_ = save_attrs_ ? 2 : 1;

// Checking the value of interp_type_
  int ocv_interp_type;
  DALI_ENFORCE(OCVInterpForDALIInterp(interp_type_, &ocv_interp_type) == DALISuccess,
               "Unknown interpolation type");
}

template <>
void Resize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  per_sample_meta_[ws->thread_idx()] = GetTransfomMeta(ws, spec_);
}

template <>
void Resize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");
  auto output = ws->Output<CPUBackend>(outputs_per_idx_ * idx);
  const auto &input_shape = input.shape();

  CheckParam(input, "Resize<CPUBackend>");

  const TransformMeta &meta = per_sample_meta_[ws->thread_idx()];

  // Resize the output & run
  output->Resize({meta.rsz_h, meta.rsz_w, meta.C});

  auto pImgInp = input.template data<uint8>();
  auto pImgOut = output->template mutable_data<uint8>();

  const auto H = input_shape[0];
  const auto W = input_shape[1];
  const auto C = input_shape[2];

  const auto cvImgType = C == 3? CV_8UC3 : CV_8UC1;
  cv::Mat inputMat(H, W, cvImgType, const_cast<unsigned char*>(pImgInp));

  // perform the resize
  cv::Mat rsz_img(meta.rsz_h, meta.rsz_w, cvImgType, const_cast<unsigned char*>(pImgOut));
  int ocv_interp_type;
  OCVInterpForDALIInterp(interp_type_, &ocv_interp_type);
  cv::resize(inputMat, rsz_img, cv::Size(meta.rsz_w, meta.rsz_h), 0, 0, ocv_interp_type);
  if (save_attrs_) {
      auto *attr_output = ws->Output<CPUBackend>(outputs_per_idx_ * idx + 1);

      attr_output->Resize(Dims{2});
      int *t = attr_output->mutable_data<int>();
      t[0] = meta.H;
      t[1] = meta.W;
    }
}

DALI_REGISTER_OPERATOR(Resize, Resize<CPUBackend>, CPU);

}  // namespace dali
