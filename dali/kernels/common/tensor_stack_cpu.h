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

#ifndef DALI_TENSOR_STACK_CPU_H
#define DALI_TENSOR_STACK_CPU_H

#include "dali/kernels/kernel.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace kernels {

enum JoinMode {
  STACK,
  CONCAT
};

namespace detail {
template<typename Out, typename In, int ndims = -1>
void StackTensors(const OutTensorCPU<Out, ndims> &out,
                  span<const InTensorCPU<In, ndims>> in, int axis) {

}


template<typename T>
void TransferBuffers(span<T> output, const TensorShape<> &output_shape,
                     span<TensorView<StorageCPU, const T>> inputs, int axis) {
  vector<int64_t> copy_sizes(inputs.size());
  for (int t = 0; t < inputs.size(); t++) {
    copy_sizes[t] = volume(inputs[t].shape.begin() + axis, inputs[t].shape.end());
  }
  auto nouter = volume(output_shape.begin(), output_shape.end());
  auto *out = output.data();
  for (ptrdiff_t outer = 0; outer < nouter; outer++) {
    for (int t = 0; t < inputs.size(); t++) {
      auto *src = inputs[t].data + outer * copy_sizes[t];
      for (ptrdiff_t inner = 0; inner < copy_sizes[t]; inner++) {
        *out++ = src[inner];
      }
    }
  }
}


template<JoinMode mode=STACK>
TensorShape<> DetermineShape(span<const TensorShape<>> in_shapes, int axis) {
  TensorShape<> ret;
  auto &insh = in_shapes[0];
  ret.resize(insh.size() + 1);
  for (int i = 0; i < axis; i++) {
    ret[i] = insh[i];
  }
  ret[axis] = in_shapes.size();
  for (int i = axis + 1; i < ret.size(); i++) {
    ret[i] = insh[i - 1];
  }
  return ret;
}


template<>
TensorShape<> DetermineShape<CONCAT>(span<const TensorShape<>> in_shapes, int axis) {
  TensorShape<> ret = in_shapes[0];
  ret[axis] *= in_shapes.size();
  return ret;
}

}  // namespace detail

template<typename Out, typename In, int ndims = -1>
struct TensorStackCpu {
  KernelRequirements Setup(KernelContext &ctx, span<const TensorShape<ndims>> in_shapes, int axis) {
    n_input_tensors_ = in_shapes.size();
    orig_shapes_ = in_shapes;
  }


  void Run(KernelContext &ctx, const OutTensorCPU<Out, ndims> &out,
           span<const InTensorCPU<In, ndims>> in, int axis) {
    DALI_ENFORCE(in.size() == n_input_tensors_, make_string(
            "Input must have the same number of tensors as was specified in call to Setup. Expected: ",
            n_input_tensors_, "Actual: ", in.size()));
    for (int i = 0; i < n_input_tensors_; i++) {
      DALI_ENFORCE(in[i].shape == orig_shapes_[i], make_string(
              "Input must have the same shapes as was specified in call to Setup. Expected: ",
              orig_shapes_[i], "Actual: ", in[i].shape));
    }


  }


  size_t n_input_tensors_;
  std::vector<TensorShape<ndims>> orig_shapes_;
};

}
}
#endif //DALI_TENSOR_STACK_CPU_H

