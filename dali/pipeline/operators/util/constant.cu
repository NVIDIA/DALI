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

#include <vector>

#include "dali/pipeline/operators/util/constant.h"

namespace dali {

template<>
void Constant<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  auto *output = ws->Output<GPUBackend>(idx);
  // Never reallocate
  output->set_num_consumers(0);
  std::vector<Dims> shape(batch_size_, source_.shape());
  output->set_type_and_size(source_.type(), shape);
  if (first_iter_[0]) {
    for (int i = 0; i < batch_size_; ++i) {
      output->Copy(source_, i,  ws->stream());
    }
    first_iter_[0] = false;
  }
}

DALI_REGISTER_OPERATOR(_Constant, Constant<GPUBackend>, GPU);

}  // namespace dali
