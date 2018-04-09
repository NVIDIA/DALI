// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
#define NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_

#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"
#include "ndll/util/image.h"

namespace ndll {

template <typename Backend>
class DumpImage : public Operator {
 public:
  explicit inline DumpImage(const OpSpec &spec) :
    Operator(spec),
    suffix_(spec.GetArgument<string>("suffix", "")) {
    NDLL_ENFORCE(spec.GetArgument<NDLLTensorLayout>("input_layout", NDLL_NHWC) == NDLL_NHWC,
        "CHW not supported yet.");
  }

  virtual inline ~DumpImage() = default;

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws, int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    NDLL_ENFORCE(input.ndim() == 3,
        "Input images must have three dimensions.");

    int h = input.dim(0);
    int w = input.dim(1);
    int c = input.dim(2);

    WriteHWCImage(input.template data<uint8>(),
        h, w, c, std::to_string(ws->data_idx()) + "-" + suffix_ + "-" + std::to_string(idx));

    // Forward the input
    output->Copy(input, 0);
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws, int idx) override {
    auto &input = ws->Input<GPUBackend>(idx);
    auto output = ws->Output<GPUBackend>(idx);

    WriteHWCBatch(input, suffix_ + "-" + std::to_string(idx));

    // Forward the input
    output->Copy(input, ws->stream());
  }

  const string suffix_;
};
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DUMP_IMAGE_H_
