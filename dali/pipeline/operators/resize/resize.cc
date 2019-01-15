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

#include <opencv2/opencv.hpp>
#include <algorithm>
#include "dali/pipeline/operators/resize/resize.h"
#include "dali/util/ocv.h"
#include "dali/kernels/static_switch.h"
#include "dali/kernels/kernel.h"

namespace dali {

using kernels::InTensorCPU;
using kernels::OutTensorCPU;

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("image_type",
        R"code(The color space of input and output image.)code", DALI_RGB)
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used.)code",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("antialias", "If true, then a filtering is applied when downscaling", false)
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
  antialias_ = spec_.GetArgument<bool>("antialias");

  // Checking the value of interp_type_
  int ocv_interp_type;
  DALI_ENFORCE(OCVInterpForDALIInterp(interp_type_, &ocv_interp_type) == DALISuccess,
               "Unknown interpolation type");
}

template <>
void Resize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  per_sample_meta_[ws->thread_idx()] = GetTransfomMeta(ws, spec_);
}

template <int channels>
void ResizeAlignCornersNearest(
    const OutTensorCPU<uint8_t, 3> &out,
    const InTensorCPU<uint8_t, 3> &in) {
  DALI_ENFORCE(in.shape[2] == channels, "Unexpected number of channels");
  DALI_ENFORCE(out.shape[2] == channels, "Unexpected number of channels");
  int h = out.shape[0];
  int w = out.shape[1];
  int srch = in.shape[0];
  int srcw = in.shape[1];

  std::vector<int> srcx(channels*w);
  for (int x = 0; x < w; x++) {
    int sx = (x + 0.5f) * srcw / w;
    if (sx < 0) x = 0;
    else if (sx >= srcw) sx = srcw - 1;
    for (int c = 0; c < channels; c++)
      srcx[channels*x + c] = channels*sx + c;
  }

  for (int y = 0; y < h; y++) {
    int sy = (y + 0.5f) * srch / h;
    if (sy < 0) y = 0;
    else if (sy >= srch) sy = srch - 1;

    auto *row_out = out(y);
    auto *row_in = in(sy);
    for (int x = 0; x < channels*w; x++) {
      row_out[x] = row_in[srcx[x]];
    }
  }
}

template <int channels>
void ResizeAlignCornersLinear(
    const OutTensorCPU<uint8_t, 3> &out,
    const InTensorCPU<uint8_t, 3> &in) {
  DALI_ENFORCE(in.shape[2] == channels, "Unexpected number of channels");
  DALI_ENFORCE(out.shape[2] == channels, "Unexpected number of channels");

  int h = out.shape[0];
  int w = out.shape[1];
  int srch = in.shape[0];
  int srcw = in.shape[1];

  std::vector<int> srcx(2*w), srcy(2*h);
  std::vector<float> inter_x(w);
  std::vector<float> inter_y(h);

  std::vector<float> tmpbuf(2*w*channels);
  struct tmprow {
    int y;
    float *data;
  };
  tmprow tmp[2] = {
    { -1, &tmpbuf[0] },
    { -1, &tmpbuf[w*channels] }
  };

  for (int x = 0; x < w; x++) {
    float fx = (x + 0.5f) * srcw / w  - 0.5f;
    int sx0 = std::max<int>(0, std::floor(fx));
    int sx1 = std::min<int>(srcw-1, std::ceil(fx));
    inter_x[x] = fx - sx0;
    srcx[2*x+0] = sx0 * channels;
    srcx[2*x+1] = sx1 * channels;
  }

  for (int y = 0; y < h; y++) {
    float fy = (y + 0.5f) * srch / h  - 0.5f;
    int sy0 = std::max<int>(0, std::floor(fy));
    int sy1 = std::min<int>(srch-1, std::ceil(fy));
    inter_y[y] = fy - sy0;
    srcy[2*y+0] = sy0;
    srcy[2*y+1] = sy1;
  }

  auto interpX = [&](float *buf, int y) {
    auto *row = in(y);
    for (int x = 0, k = 0; x < w; x++) {
      auto *in0 = &row[srcx[2*x]];
      auto *in1 = &row[srcx[2*x+1]];
      float q = inter_x[x];
      for (int c = 0; c < channels; c++, k++) {
        buf[k] = in0[c] + (in1[c] - in0[c]) * q;
      }
    }
  };

  for (int y = 0; y < h; y++) {
    int sy0 = srcy[2*y];
    int sy1 = srcy[2*y+1];

    auto *src0 = in(sy0);
    auto *src1 = in(sy1);

    float *row0, *row1;
    if (sy0 == tmp[0].y) {
      row0 = tmp[0].data;
    } else if (sy0 == tmp[1].y) {
      row0 = tmp[1].data;
    } else {
      int tmpidx = (sy1 == tmp[0].y) ? 1 : 0;
      row0 = tmp[tmpidx].data;
      tmp[tmpidx].y = sy0;
      interpX(row0, sy0);
    }

    if (sy1 == tmp[0].y) {
      row1 = tmp[0].data;
    } else if (sy1 == tmp[1].y) {
      row1 = tmp[1].data;
    } else {
      int tmpidx = (sy0 == tmp[0].y) ? 1 : 0;
      row1 = tmp[tmpidx].data;
      tmp[tmpidx].y = sy1;
      interpX(row1, sy1);
    }

    auto *row = out(y);
    float q = inter_y[y];
    for (int x = 0; x < w * channels; x++) {
      row[x] = row0[x] + (row1[x] - row0[x]) * q;
    }
  }
}

template <int channels>
void ResizeAlignCorners(
    const OutTensorCPU<uint8_t, 3> &out,
    const InTensorCPU<uint8_t, 3> &in,
    DALIInterpType interp) {
  DALI_ENFORCE(in.shape[2] == channels, "Unexpected number of channels");
  DALI_ENFORCE(out.shape[2] == channels, "Unexpected number of channels");
  if (interp == DALI_INTERP_LINEAR)
    ResizeAlignCornersLinear<channels>(out, in);
  else if (interp == DALI_INTERP_NN)
    ResizeAlignCornersNearest<channels>(out, in);
  else
    DALI_FAIL("Unsupported interpolation type");
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

  InTensorCPU<uint8_t, 3> in(pImgInp, { H, W, C });
  OutTensorCPU<uint8_t, 3> out(pImgOut, { meta.rsz_h, meta.rsz_w, meta.C });

  if (interp_type_ == DALI_INTERP_NN) {
    VALUE_SWITCH(C, channels, (1, 2, 3, 4),
      (ResizeAlignCornersNearest<channels>(out, in);)
    , DALI_FAIL("Unsupported number of channels: " + std::to_string(C)));
  } else  {
    const auto cvImgType = C == 3? CV_8UC3 : CV_8UC1;
    cv::Mat inputMat(H, W, cvImgType, const_cast<unsigned char*>(pImgInp));

    const float threshold = 0.8f;
    if ((meta.rsz_w < threshold*W || meta.rsz_h < threshold*H) && antialias_) {
      cv::Mat tmp(H, W, cvImgType);
      auto sigma=[](float src, float dst)->std::pair<int, float> {
        float s = 0.6f * src / dst - 0.25f;
        if (s <= 1e-3)
          return { 1, 1e-3 };
        int n = 2*ceil(s)+1;
        return { n, s };
      };
      auto fx = sigma(W, meta.rsz_w);
      auto fy = sigma(H, meta.rsz_h);
      cv::GaussianBlur(inputMat, tmp, {fx.first, fy.first}, fx.second, fy.second, CV_HAL_BORDER_REFLECT_101);
      inputMat = std::move(tmp);
    }

    // perform the resize
    cv::Mat rsz_img(meta.rsz_h, meta.rsz_w, cvImgType, const_cast<unsigned char*>(pImgOut));
    int ocv_interp_type;
    OCVInterpForDALIInterp(interp_type_, &ocv_interp_type);
    cv::resize(inputMat, rsz_img, cv::Size(meta.rsz_w, meta.rsz_h), 0, 0, ocv_interp_type);
  }
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
