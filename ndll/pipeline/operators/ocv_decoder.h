// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_OCV_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_OCV_DECODER_H_

#include <opencv2/opencv.hpp>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class OCVDecoder : public Operator {
 public:
  explicit inline OCVDecoder(const OpSpec &spec) :
    Operator(spec),
    output_type_(spec.GetArgument<NDLLImageType>("output_type", NDLL_RGB)),
    c_(IsColor(output_type_) ? 3 : 1) {}

  virtual inline ~OCVDecoder() = default;
  DISABLE_COPY_MOVE_ASSIGN(OCVDecoder);

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    // Verify input
    NDLL_ENFORCE(input.ndim() == 1,
        "Input must be 1D encoded jpeg string.");
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Input must be stored as uint8 data.");

    // Decode image to tmp cv::Mat
    cv::Mat tmp = cv::imdecode(
        cv::Mat(1, input.size(), CV_8UC1, const_cast<void*>(input.raw_data())),
        IsColor(output_type_) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    // if RGB needed, permute from BGR
    if (output_type_ == NDLL_RGB) {
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

  NDLLImageType output_type_;
  int c_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_OCV_DECODER_H_
