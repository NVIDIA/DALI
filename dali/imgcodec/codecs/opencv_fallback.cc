// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/codecs/opencv_fallback.h"
#include "dali/imgcodec/util/convert.h"

#include <opencv2/imgcodecs.hpp>

namespace dali {
namespace imgcodec {

DecodeResult OpenCVCodecInstance::Decode(SampleView<CPUBackend> out,
                                         ImageSource *in,
                                         DecodeParams opts) {
  int flags = 0;

  switch (opts.format) {
    case DALI_ANY_DATA:
      // Note: IMREAD_UNCHANGED always ignores orientation
      flags |= cv::IMREAD_UNCHANGED;
      break;
    case DALI_GRAY:
      flags |= cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION;
      break;
    default:
      flags |= cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION;
  }

  if (opts.dtype != DALI_UINT8)
    flags |= cv::IMREAD_ANYDEPTH;

  DecodeResult res;
  #ifndef _DEBUG
  try {
  #endif
    cv::Mat cvimg;
    if (in->GetKind() == InputKind::Filename) {
      cvimg = cv::imread(in->GetFilename(), flags);
    } else {
      assert(in->GetKind() == InputKind::HostMemory);
      auto *raw = static_cast<const uint8_t *>(in->GetRawData());
      cvimg = cv::imdecode(cv::_InputArray(raw, in->GetSize()), flags);
    }
    res.success = cvimg.ptr(0) != nullptr;
    if (res.success) {
      // TODO(michalz): invoke appropriate copy/conversion
      // Convert(out, cvimg, opt);
    }
  #ifndef _DEBUG
  } catch (...) {
    res.exception = std::current_exception();
    res.success = false;
  }
  #endif

  return res;
}


}  // namespace imgcodec
}  // namespace dali
