// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/copy.h"

namespace dali {

template<>
void Copy<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);
  output->set_type(input.type());
  output->ResizeLike(input);

  TypeInfo type = input.type();
  type.Copy<CPUBackend, CPUBackend>(
      output->raw_mutable_data(),
      input.raw_data(), input.size(), 0);
}

DALI_REGISTER_OPERATOR(Copy, Copy<CPUBackend>, CPU);

DALI_SCHEMA(Copy)
  .DocStr("Make a copy of the input tensor")
  .NumInput(1)
  .NumOutput(1);

}  // namespace dali
