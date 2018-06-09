// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "ndll/pipeline/operators/util/cast.h"

namespace ndll {

template<>
void Cast<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  NDLLDataType itype = input.type().id();

  NDLL_TYPE_SWITCH(output_type_, OType,
      output->mutable_data<OType>();
      output->ResizeLike(input);
      NDLL_TYPE_SWITCH(itype, IType,
        CPUHelper<IType, OType>(
          output->mutable_data<OType>(),
          input.data<IType>(),
          input.size());););
}

NDLL_REGISTER_OPERATOR(Cast, Cast<CPUBackend>, CPU);

NDLL_SCHEMA(Cast)
  .DocStr("Cast tensor to a different type")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("dtype",
      R"code(`ndll.types.NDLLDataType`
      Output data type)code");

}  // namespace ndll
