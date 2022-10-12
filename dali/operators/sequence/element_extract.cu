// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include "dali/operators/sequence/element_extract.h"

namespace dali {

template <>
void ElementExtract<GPUBackend>::RunCopies(Workspace &ws) {
  scatter_gather_.Run(ws.stream(), true);
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<GPUBackend>, GPU);

}  // namespace dali
