// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/transpose/transpose.h"
#include "dali/error_handling.h"

namespace dali {

#define cuttCheck(stmt) do {                                   \
  cuttResult err = stmt;                                       \
  if (err != CUTT_SUCCESS) {                                   \
    DALI_FAIL("Error while transposing" + std::string(#stmt)); \
  }                                                            \
} while(0)

void Transpose::~Transpose() {
  if (cutt_handle_ > -1) {
    cuttCheck(cuttDestroy(plan));
  }
}


void Transpose::NaiveTransposeKernel(const TensorList<GPUBackend>& input, TensorList<GPUBackend>* output) {

}

template <typename T = int>
void Transpose::cuTTKernel(const TensorList<GPUBackend>& input, TensorList<GPUBackend>* output) {
  std::vector<Index> input_shape = input.tensor_shape(0);
  // TODO copy and cast here vv
  const int *dim = input_shape.data();
  const int *permutation = perm_.data();

  if (cutt_handle_ == -1) {
    cuttCheck(cuttPlan(&cutt_handle_, perm_.size(), dim, permutation, sizeof(T), 0));
  }

  for (int i = 0; i < batch_size; ++i) {
    const T* in = input.tensor<T>(i);
    T* out = input.mutable_tensor<T>(i);
    cuttCheck(cuttExecute(cutt_handle_, in, out));
  }
}



template<>
void Transpose<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  auto *output = ws->Output<GPUBackend>(idx);

  std::vector<Index> input_shape = input.tensor_shape(0);
  DALI_ENFORCE(input_shape.size() == perm_.size(), "Transposed tensors rank should be equal to the permutation index list.");
  // TODO enforce perm indices to be [0, n]

  if (input.IsDenseTensor()) {
    cuTTKernel(input, output);
  } else {
    NaiveTransposeKernel(input, output);
  }
}

DALI_REGISTER_OPERATOR(Transpose, Transpose<GPUBackend>, GPU);

}  // namespace dali

