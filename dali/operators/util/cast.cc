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


#include "dali/core/static_switch.h"
#include "dali/operators/util/cast.h"

namespace dali {

template<>
void Cast<CPUBackend>::RunImpl(SampleWorkspace &ws) {
  auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);

  DALIDataType itype = input.type().id();

  TYPE_SWITCH(output_type_, type2id, OType, CAST_ALLOWED_TYPES, (
    output.mutable_data<OType>();
    output.ResizeLike(input);
    TYPE_SWITCH(itype, type2id, IType, CAST_ALLOWED_TYPES, (
      CPUHelper<OType, IType>(output.mutable_data<OType>(), input.data<IType>(), input.size());
    ), DALI_FAIL("Invalid input type"););  // NOLINT(whitespace/parens)
  ), DALI_FAIL("Invalid output type"););  // NOLINT(whitespace/parens)
}

DALI_REGISTER_OPERATOR(Cast, Cast<CPUBackend>, CPU);

DALI_SCHEMA(Cast)
  .DocStr("Cast tensor to a different type")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("dtype",
      R"code(Output data type.)code",
      DALI_DATA_TYPE);

}  // namespace dali
