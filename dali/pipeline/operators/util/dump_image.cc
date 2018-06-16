// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/dump_image.h"
#include "dali/util/image.h"

namespace dali {

template<>
void DumpImage<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  DALI_ENFORCE(input.ndim() == 3,
      "Input images must have three dimensions.");

  int h = input.dim(0);
  int w = input.dim(1);
  int c = input.dim(2);

  WriteHWCImage(input.template data<uint8>(),
      h, w, c, std::to_string(ws->data_idx()) + "-" + suffix_ + "-" + std::to_string(idx));

  // Forward the input
  output->Copy(input, 0);
}

DALI_REGISTER_OPERATOR(DumpImage, DumpImage<CPUBackend>, CPU);

DALI_SCHEMA(DumpImage)
  .DocStr(R"code(Save images in batch to disk in PPM format.
  Useful for debugging.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalArg("suffix",
      R"code(`string`
      Suffix to be added to output file names)code", "")
  .AddOptionalArg("input_layout",
      R"code(`dali.types.DALITensorLayout`
      Layout of input images)code", DALI_NHWC);

}  // namespace dali
