// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
#define NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/jpeg.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class TJPGDecoder : public Operator {
 public:
  explicit inline TJPGDecoder(const OpSpec &spec) :
    Operator(spec),
    output_type_(spec.GetArgument<NDLLImageType>("output_type", NDLL_RGB)),
    c_(IsColor(output_type_) ? 3 : 1) {}

  virtual inline ~TJPGDecoder() = default;
  DISABLE_COPY_MOVE_ASSIGN(TJPGDecoder);

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    // Verify input
    NDLL_ENFORCE(input.ndim() == 1,
        "Input must be 1D encoded jpeg string.");
    NDLL_ENFORCE(IsType<uint8>(input.type()),
        "Input must be stored as uint8 data.");

    NDLL_CALL(DecodeJPEGHost(
            input.template data<uint8>(),
            input.size(), output_type_,
            output));
  }

  NDLLImageType output_type_;
  int c_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_TJPG_DECODER_H_
