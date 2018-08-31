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


#include "dali/pipeline/operators/util/constant.h"

namespace dali {

template<>
void Constant<CPUBackend>::RunImpl(SampleWorkspace *ws, int idx) {
  auto *output = ws->Output<CPUBackend>(idx);
  int sample_idx = ws->data_idx();
  // Never reallocate
  output->set_num_consumers(0);
  if (first_iter_[sample_idx]) {
    output->Copy(source_, 0);
    first_iter_[sample_idx] = false;
  }
}

DALI_REGISTER_OPERATOR(_Constant, Constant<CPUBackend>, CPU);

DALI_SCHEMA(_Constant)
  .DocStr("Return constant tensor.")
  .NumInput(0)
  .NumOutput(1)
  .AddArg("source_dtype",
      R"code(Type of tensor.)code",
      DALI_DATA_TYPE)
  .AddArg("source_shape",
      R"code(Shape of tensor.)code",
      DALI_INT_VEC)
  .AddArg("source_data",
      R"code(Tensor data.)code",
      DALI_VOID_PTR);

DALI_SCHEMA(Constant)
  .DocStr("Return constant tensor.")
  .NumInput(0)
  .NumOutput(1)
  .AddArg("source",
      R"code(Tensor to return.)code",
      DALI_NUMPY_BUF);
}  // namespace dali
