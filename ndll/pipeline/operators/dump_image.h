// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
#define NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_

#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename Backend>
class DumpImage : public Operator<Backend> {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator<Backend>(spec),
    suffix_(spec.GetArgument<string>("suffix", "")),
    hwc_(spec.GetArgument<bool>("hwc_format", true)) {
    NDLL_ENFORCE(hwc_, "CHW not supported yet.");
  }

  virtual inline ~DumpImage() = default;

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto output = ws->Output<CPUBackend>(0);

    NDLL_ENFORCE(input.ndim() == 3,
        "Input images must have three dimensions.");

    int h = input.dim(0);
    int w = input.dim(1);
    int c = input.dim(2);

    WriteHWCImage(input.template data<uint8>(),
        h, w, c, std::to_string(ws->data_idx()) + "-" + suffix_);

    // Forward the input
    output->Copy(input, 0);
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto output = ws->Output<CPUBackend>(0);

    WriteHWCBatch(input, suffix_);

    // Forward the input
    output->Copy(input, ws->stream());
  }

  const string suffix_;
  bool hwc_;
};
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
