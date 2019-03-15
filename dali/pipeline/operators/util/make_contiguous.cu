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

#include "dali/pipeline/operators/util/make_contiguous.h"

namespace dali {

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguous, Mixed);

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation, more suitable for execution on the GPU)code")
  .NumInput(1)
  .NumOutput(1)
  .MakeInternal();

}  // namespace dali
