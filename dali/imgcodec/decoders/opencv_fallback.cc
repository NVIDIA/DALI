// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "dali/imgcodec/decoders/opencv_fallback.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/parsers/jpeg.h"

namespace dali {
namespace imgcodec {

inline DALIDataType CV2DaliType(int cv_type_id) {
  switch (cv_type_id) {
    case CV_8U:
      return DALI_UINT8;
    case CV_8S:
      return DALI_INT8;
    case CV_16U:
      return DALI_UINT16;
    case CV_16S:
      return DALI_INT16;
    case CV_32S:
      return DALI_INT32;
    case CV_32F:
      return DALI_FLOAT;
    case CV_64F:
      return DALI_FLOAT64;
    default:
      throw std::invalid_argument(make_string("Unexpected OpenCV type: ", cv_type_id));
  }
}

DecodeResult OpenCVDecoderInstance::DecodeImplTask(int thread_idx,
                                                   SampleView<CPUBackend> out,
                                                   ImageSource *in,
                                                   DecodeParams opts,
                                                   const ROI &roi) {
  (void) thread_idx;  // this implementation doesn't use per-thread resources
  int flags = 0;
  bool adjust_orientation = false;
  DALIImageType in_format;

  switch (opts.format) {
    case DALI_ANY_DATA:
      // Note: IMREAD_UNCHANGED always ignores orientation
      flags |= cv::IMREAD_UNCHANGED;
      adjust_orientation = opts.use_orientation;
      in_format = DALI_ANY_DATA;
      break;

    case DALI_GRAY:
      flags |= cv::IMREAD_GRAYSCALE;
      if (!opts.use_orientation)
        flags |= cv::IMREAD_IGNORE_ORIENTATION;
      in_format = DALI_GRAY;
      break;

    default:
      flags |= cv::IMREAD_COLOR;
      if (!opts.use_orientation)
        flags |= cv::IMREAD_IGNORE_ORIENTATION;
      in_format = DALI_BGR;
      break;
  }

  if (opts.dtype != DALI_UINT8)
    flags |= cv::IMREAD_ANYDEPTH;

  DecodeResult res;
  try {
    cv::Mat cvimg;
    if (in->Kind() == InputKind::Filename) {
      cvimg = cv::imread(in->Filename(), flags);
    } else {
      assert(in->Kind() == InputKind::HostMemory);
      const auto *raw = static_cast<const uint8_t *>(in->RawData());
      cvimg = cv::imdecode(cv::_InputArray(raw, in->Size()), flags);
    }

    Orientation orientation = {};
    if (adjust_orientation) {
      auto info = ImageFormatRegistry::instance().GetImageFormat(in)->Parser()->GetInfo(in);
      orientation = info.orientation;
    }

    res.success = cvimg.ptr(0) != nullptr;
    if (res.success) {
      DALI_ENFORCE(cvimg.dims + 1 == out.shape().sample_dim(), make_string(
        "The decoded image has an unexpected number of spatial dimensions: ", cvimg.dims,
        "\nExpected: ", out.shape().sample_dim() - 1));

      DALIDataType type = CV2DaliType(cvimg.depth());

      TensorShape<> shape;
      shape.resize(cvimg.dims + 1);
      for (int d = 0; d < cvimg.dims; d++)
        shape[d] = cvimg.size[d];
      shape[cvimg.dims] = cvimg.channels();

      SampleView<CPUBackend> in(cvimg.ptr(0), shape, type);

      int in_channels = cvimg.channels();
      auto out_format = opts.format;
      if (out_format == DALI_ANY_DATA && in_format == DALI_ANY_DATA) {
        if (in_channels == 3) {
          // OpenCV uses BGR by default. Here we avoid outputting BGR when requesting ANY_DATA
          in_format = DALI_BGR;
          out_format = DALI_RGB;
        }  // TODO(michalz): support RGBA in DALI
      }
      TensorLayout layout = cvimg.dims == 3 ? "DHWC" : "HWC";

      Convert(out, layout, out_format, in, layout, in_format, roi, orientation);
    } else {
      JpegParser jpeg_parser{};
      if (jpeg_parser.CanParse(in)) {
        auto ext_info = jpeg_parser.GetExtendedInfo(in);
        std::array<uint8_t, 2> sof3_marker = {0xff, 0xc3};
        if (ext_info.sof_marker == sof3_marker) {
          res.exception = std::make_exception_ptr(
            std::runtime_error(
              make_string(
                "Failed to decode a JPEG lossless (SOF-3) sample: ", in->SourceInfo(), ".\n"
                "Support for lossless is currently very limited:\n"
                "- Only number of components <= 2\n"
                "- Only predictor value 1\n"
                "- Only \"mixed\" backend\n"
                "If you used the \"cpu\" backend, please try the \"mixed\" one instead\n")));
        }
      }
    }
  } catch (...) {
    res.exception = std::current_exception();
    res.success = false;
  }

  return res;
}

REGISTER_DECODER("*", OpenCVDecoderFactory, FallbackDecoderPriority);

}  // namespace imgcodec
}  // namespace dali
