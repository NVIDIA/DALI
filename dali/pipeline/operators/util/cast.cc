// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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
      R"code(Output data type.)code",
      DALI_DATA_TYPE);

}  // namespace dali
