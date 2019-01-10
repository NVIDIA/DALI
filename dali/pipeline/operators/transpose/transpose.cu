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

#include <chrono>
#include <string>
#include <vector>

#include "dali/pipeline/operators/transpose/transpose.h"
#include "dali/error_handling.h"

namespace dali {

#define cuttCheck(stmt) do {                                   \
  cuttResult err = stmt;                                       \
  if (err != CUTT_SUCCESS) {                                   \
    DALI_FAIL("Error while transposing" + std::string(#stmt)); \
  }                                                            \
} while (0)

template <>
Transpose<GPUBackend>::~Transpose() {
  if (cutt_handle_ > 0) {
    cuttCheck(cuttDestroy(cutt_handle_));
  }
}


template <>
void Transpose<GPUBackend>::NaiveTransposeKernel(const TensorList<GPUBackend>& input,
                                     TensorList<GPUBackend>* output) {
  DALI_FAIL("NaiveTransposeKernel not implemented yet.");
}

template <>
template <typename T>
void Transpose<GPUBackend>::cuTTKernel(const TensorList<GPUBackend>& input,
                           TensorList<GPUBackend>* output,
                           cudaStream_t stream) {
  Dims tmp = input.tensor_shape(0);
  std::vector<int> input_shape(tmp.begin(), tmp.end());

  int *dim = const_cast<int*>(input_shape.data());
  int *permutation = const_cast<int*>(perm_.data());

  if (cutt_handle_ == 0) {
    cuttCheck(cuttPlan(&cutt_handle_, perm_.size(), dim, permutation, sizeof(T), stream));
  }

  for (int i = 0; i < batch_size_; ++i) {
    const void* in = input.raw_tensor(i);
    void* out = output->raw_mutable_tensor(i);
    cuttCheck(cuttExecute(cutt_handle_, in, out));
  }
}

template <>
template <typename T>
void Transpose<GPUBackend>::cuTTKernelBatched(const TensorList<GPUBackend>& input,
                           TensorList<GPUBackend>* output,
                           cudaStream_t stream) {
  Dims tmp = input.tensor_shape(0);
  std::vector<int> input_shape(tmp.begin(), tmp.end());
  input_shape.insert(input_shape.begin(), batch_size_);

  std::vector<int> batched_perm(perm_.begin(), perm_.end());
  std::for_each(batched_perm.begin(), batched_perm.end(), [](int& i) { i++; });
  batched_perm.insert(batched_perm.begin(), 0);

  int *dim = const_cast<int*>(input_shape.data());
  int *permutation = const_cast<int*>(batched_perm.data());

  if (cutt_handle_ == 0) {
    cuttCheck(cuttPlan(&cutt_handle_, batched_perm.size(), dim, permutation, sizeof(T), stream));
  }

  const void* in = input.raw_tensor(0);
  void* out = output->raw_mutable_tensor(0);
  cuttCheck(cuttExecute(cutt_handle_, in, out));
}

template<>
void Transpose<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  auto &input = ws->Input<GPUBackend>(0);
  auto* tl_sequence_output = ws->Output<GPUBackend>(0);
  tl_sequence_output->set_type(TypeInfo::Create<float>());
  // tl_sequence_output->SetLayout(DALI_UNKNOWN);
}


inline Dims GetPermutedDims(const Dims& dims, const std::vector<int>& permutation) {
  Dims permuted_dims;
  for (auto idx : permutation) {
    permuted_dims.push_back(dims[idx]);
  }
  return permuted_dims;
}

template<>
void Transpose<GPUBackend>::RunImpl(DeviceWorkspace* ws, int idx) {
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

  const auto& input = ws->Input<GPUBackend>(idx);
  auto* output = ws->Output<GPUBackend>(idx);

  Dims input_shape = input.tensor_shape(0);
  DALI_ENFORCE(input_shape.size() == perm_.size(),
               "Transposed tensors rank should be equal to the permutation index list.");

  if (input.IsDenseTensor()) {
    // Actually we can work directly on the TL here
    Dims permuted_dims = GetPermutedDims(input_shape, perm_);
    output->Resize(std::vector<Dims>(batch_size_, permuted_dims));
    if (batch_size_ > 1)
      cuTTKernelBatched(input, output, ws->stream());
    else
      cuTTKernel(input, output, ws->stream());
  } else {
    std::vector<Dims> tl_shape;
    for (int i = 0; i < batch_size_; ++i) {
      Dims in_shape = input.tensor_shape(i);
      tl_shape.emplace_back(GetPermutedDims(in_shape, perm_));
    }
    output->Resize(tl_shape);
    NaiveTransposeKernel(input, output);
  }

  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast< std::chrono::duration<double> >
                    (end - start).count();
  std::cout << "Transpose duration: " << (duration*1000.0) << " ms" << std::endl;
}

DALI_REGISTER_OPERATOR(Transpose, Transpose<GPUBackend>, GPU);

}  // namespace dali
