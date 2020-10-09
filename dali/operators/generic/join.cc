// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/generic/join.h"

namespace dali {

template <typename Backend, bool new_axis>
void TensorJoin<Backend, new_axis>::CollectInputs(const workspace_t<Backend> &ws) {
  int njoin = ws.NumRegularInput()
  inputs_.resize(njoin);
  output_type_ = ws.template InputRef<Backend>(i).type();
  for (int i = 1; i < njoin; i++) {
    DALI_ENFORCE(ws.template InputRef<Backend>(i).type().id() == output_type_.
  }
}


}  // namespace dali
