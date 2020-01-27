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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/operators/generic/transpose/transpose.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"

namespace dali {

#define cuttCheck(stmt) do {                                   \
  cuttResult err = stmt;                                       \
  if (err != CUTT_SUCCESS) {                                   \
    DALI_FAIL("Error while transposing " + std::string(#stmt)); \
  }                                                            \
} while (0)

namespace kernel {

template <typename T>
void cuTTKernel(const TensorList<GPUBackend>& input,
                TensorList<GPUBackend>& output,
                const std::vector<int>& permutation,
                cudaStream_t stream) {
  int batch_size = static_cast<int>(input.ntensor());
  for (int i = 0; i < batch_size; ++i) {
    transpose_detail::VecInt shape;
    for (auto s : input.tensor_shape(i))
      shape.push_back(s);

    transpose_detail::VecInt perm;
    for (auto p : permutation)
      perm.push_back(p);

    transpose_detail::PrepareArguments(shape, perm, true);

    const void* in = input.raw_tensor(0);
    void* out = output.raw_mutable_tensor(0);
    cuttHandle plan;
    cuttCheck(cuttPlan(&plan,
                       shape.size(),
                       shape.data(),
                       perm.data(),
                       sizeof(T),
                       stream));
    cuttCheck(cuttExecute(plan, in, out));
    CUDA_CALL(cudaStreamSynchronize(stream));
    cuttCheck(cuttDestroy(plan));
  }
}

template <typename T>
void cuTTKernelBatched(const TensorList<GPUBackend>& input,
                       TensorList<GPUBackend>& output,
                       const std::vector<int>& permutation,
                       cuttHandle* plan,
                       cudaStream_t stream) {
  int batch_size = static_cast<int>(input.ntensor());
  transpose_detail::VecInt shape;
  shape.push_back(batch_size);
  for (auto &s : input.tensor_shape(0))
    shape.push_back(s);

  transpose_detail::VecInt perm;
  perm.push_back(0);
  for (auto p : permutation)
    perm.push_back(p+1);

  transpose_detail::PrepareArguments(shape, perm, true);

  if (*plan == 0) {
    cuttCheck(cuttPlan(plan,
                       shape.size(),
                       shape.data(),
                       perm.data(),
                       sizeof(T),
                       stream));
  }

  const void* in = input.raw_tensor(0);
  void* out = output.raw_mutable_tensor(0);
  cuttCheck(cuttExecute(*plan, in, out));
}
}  // namespace kernel


class TransposeGPU : public Transpose<GPUBackend> {
 public:
  explicit inline TransposeGPU(const OpSpec &spec) : Transpose(spec) {}

  void RunImpl(DeviceWorkspace &ws) override {
    const auto& input = ws.Input<GPUBackend>(0);
    auto& output = ws.Output<GPUBackend>(0);

    TypeInfo itype = input.type();
    DALI_ENFORCE((itype.size() == 1 || itype.size() == 2 || itype.size() == 4 || itype.size() == 8),
        "cuTT transpose supports only [1-2-4-8] bytes types.");

    output.SetLayout(output_layout_);

    auto input_shape = input.tensor_shape(0);
    DALI_ENFORCE(input_shape.size() == static_cast<int>(perm_.size()),
                "Transposed tensors rank should be equal to the permutation index list.");

    if (input.IsDenseTensor()) {
      if (cutt_handle_ != 0) {
        if (input_shape != previous_iter_shape_) {
          cuttCheck(cuttDestroy(cutt_handle_));
          cutt_handle_ = 0;
          previous_iter_shape_ = input_shape;
        }
      } else {
        previous_iter_shape_ = input_shape;
      }
      if (itype.size() == 1) {
        kernel::cuTTKernelBatched<uint8_t>(input, output, perm_, &cutt_handle_, ws.stream());
      } else if (itype.size() == 2) {
        kernel::cuTTKernelBatched<uint16_t>(input, output, perm_, &cutt_handle_, ws.stream());
      } else if (itype.size() == 4) {
        kernel::cuTTKernelBatched<int32_t>(input, output, perm_, &cutt_handle_, ws.stream());
      } else {  // itype.size() == 8
        kernel::cuTTKernelBatched<int64_t>(input, output, perm_, &cutt_handle_, ws.stream());
      }
    } else {
      if (itype.size() == 1) {
        kernel::cuTTKernel<uint8_t>(input, output, perm_, ws.stream());
      } else if (itype.size() == 2) {
        kernel::cuTTKernel<uint16_t>(input, output, perm_, ws.stream());
      } else if (itype.size() == 4) {
        kernel::cuTTKernel<int32_t>(input, output, perm_, ws.stream());
      } else {  // itype.size() == 8
        kernel::cuTTKernel<int64_t>(input, output, perm_, ws.stream());
      }
    }
  }

  ~TransposeGPU() {
    if (cutt_handle_ > 0) {
      auto err = cuttDestroy(cutt_handle_);
      if (err != CUTT_SUCCESS) {
        // Something terrible happened. Just quit now, before you'll loose your life or worse...
        std::terminate();
      }
    }
  }

 private:
  cuttHandle cutt_handle_ = 0;
  // used by dense TL cuttHandle
  TensorShape<> previous_iter_shape_;
};

DALI_REGISTER_OPERATOR(Transpose, TransposeGPU, GPU);

}  // namespace dali
