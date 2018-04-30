// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/dump_image.h"
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

NDLL_OPERATOR_SCHEMA(DumpImage)
  .DocStr("Save images in batch to disk in PPM format")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("suffix", "Suffix to be added to output file names", "")
  .AddOptionalArg("input_layout", "Layout of input images", NDLL_NHWC);

}  // namespace ndll
