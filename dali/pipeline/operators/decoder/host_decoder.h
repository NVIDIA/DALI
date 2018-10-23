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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_

#include <opencv2/opencv.hpp>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/image/jpeg.h"
#include "dali/image/png.h"

namespace dali {

class HostDecoder : public Operator<CPUBackend> {
 public:
  explicit inline HostDecoder(const OpSpec &spec) :
    Operator<CPUBackend>(spec),
    output_type_(spec.GetArgument<DALIImageType>("output_type")),
    c_(IsColor(output_type_) ? 3 : 1) {}

  virtual inline ~HostDecoder() = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoder);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    // Verify input
    DALI_ENFORCE(input.ndim() == 1,
        "Input must be 1D encoded jpeg string.");
    DALI_ENFORCE(IsType<uint8>(input.type()),
        "Input must be stored as uint8 data.");

    // determine what format the image is.
    if (CheckIsJPEG(input.data<uint8>(), input.size())) {
      // JPEG: Pass to TurboJPEG-based decode
      DALI_CALL_EX(DecodeJPEGHost(input.data<uint8>(),
                               input.size(),
                               output_type_,
                               output),
                              "Problem with file: " + input.GetSourceInfo());
    } else if (CheckIsPNG(input.data<uint8>(), input.size())) {
      // PNG: Pass to OCV-based decode without extra copy
      DALI_CALL_EX(DecodePNGHost(input.data<uint8>(),
                              input.size(),
                              output_type_,
                              output),
                              "Problem with file: " + input.GetSourceInfo());
    } else {
      // all other cases use openCV (for now)

      // Decode image to tmp cv::Mat
      cv::Mat tmp = cv::imdecode(
          cv::Mat(1, input.size(), CV_8UC1, const_cast<void*>(input.raw_data())),
          IsColor(output_type_) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

      // if RGB needed, permute from BGR
      if (output_type_ == DALI_RGB) {
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
      }

      // Resize actual storage
      const int W = tmp.cols;
      const int H = tmp.rows;
      output->Resize({H, W, c_});
      unsigned char *out_data = output->mutable_data<unsigned char>();

      // Copy to final output
      std::memcpy(out_data,
                  tmp.ptr(),
                  H*W*c_);
    }
  }

  DALIImageType output_type_;
  int c_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_
