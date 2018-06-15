// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/util/dump_image.h"
#include "ndll/util/image.h"

namespace ndll {

template<>
void DumpImage<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
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

NDLL_REGISTER_OPERATOR(DumpImage, DumpImage<CPUBackend>, CPU);

NDLL_SCHEMA(DumpImage)
  .DocStr(R"code(Save images in batch to disk in PPM format.
  Useful for debugging.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("suffix",
      R"code(`string`
      Suffix to be added to output file names)code", "")
  .AddOptionalArg("input_layout",
      R"code(`ndll.types.NDLLTensorLayout`
      Layout of input images)code", NDLL_NHWC);

}  // namespace ndll
