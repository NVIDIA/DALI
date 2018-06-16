// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include "dali/pipeline/operators/util/cast.h"

namespace dali {

template<>
void Cast<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  auto *output = ws->Output<CPUBackend>(idx);

  DALIDataType itype = input.type().id();

  DALI_TYPE_SWITCH(output_type_, OType,
      output->mutable_data<OType>();
      output->ResizeLike(input);
      DALI_TYPE_SWITCH(itype, IType,
        CPUHelper<IType, OType>(
          output->mutable_data<OType>(),
          input.data<IType>(),
          input.size());););
}

DALI_REGISTER_OPERATOR(Cast, Cast<CPUBackend>, CPU);

DALI_SCHEMA(Cast)
  .DocStr("Cast tensor to a different type")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("dtype",
      R"code(`dali.types.DALIDataType`
      Output data type)code");

}  // namespace dali
