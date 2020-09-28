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
#include "dali/core/format.h"

namespace dali {
namespace kernels {

enum JoinMode {
  STACK,
  CONCAT
};

namespace detail {

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

template<typename Out, typename In, JoinMode mode = STACK, int dims = -1>
struct TensorJoinCpu {
  KernelRequirements Setup(KernelContext &ctx, span<const TensorShape<dims>> in_shapes, int axis) {
    n_input_tensors_ = in_shapes.size();
    orig_shapes_(in_shapes);
    auto ndims = in_shapes[0].sample_dim();
    DALI_ENFORCE(axis < ndims && axis > -ndims, "Incorrect axis. Actual: ", axis, ". Expected in [",
                 -ndims + 1, ", ", ndims - 1, "] interval");
    axis_ = axis >= 0 ? axis : ndims + axis;

    {
      const auto &ref = in_shapes[0];
      for (int i = 1; i < n_input_tensors_; i++) {
        DALI_ENFORCE(in_shapes[i].sample_dim() == ref.sample_dim(),
                     "Every input shape must have the same number of dimensions.");
        for (int j = 0; j < ref.size(); j++) {
          if (mode == CONCAT) {
            DALI_ENFORCE(in_shapes[i][j] == ref.shape[j] || (j == axis_ && mode == CONCAT),
                         make_string(
                                 "Number of samples in every dimension "
                                 "(but the one along which concatenation occurs) must be the same "
                                 "(CONCAT mode). 0-th shape at index ", j, " has dimension ",
                                 ref.shape[j], ", while ", i, "-th shape at index ", j,
                                 " has dimension ", in_shapes[i][j]));
          } else {
            DALI_ENFORCE(in_shapes[i][j] == ref.shape[j], make_string(
                    "Number of samples in every dimension must be the same (STACK mode). "
                    "0-th shape at index ", j, " has dimension ", ref.shape[j], ", while ", i,
                    "-th shape at index ", j, " has dimension ", in_shapes[i][j]));
          }
        }
      }
    }

    KernelRequirements kr;
    output_shape_ = detail::DetermineShape<mode>(in_shapes, axis);
    kr.output_shapes.resize(1);
    TensorListShape<> tmp({output_shape_});  // clang's destructor bug still haunting
    kr.output_shapes[0] = tmp;
    return kr;
  }


  KernelRequirements Setup(KernelContext &ctx, span<const InTensorCPU<In, dims>> in, int axis) {
    std::vector<TensorShape<>> in_shapes(in.size());
    for (int i = 0; i < in.size(); i++) {
      in_shapes[i] = in[i].shape;
    }
    return Setup(ctx, in_shapes, axis);
  }


  void Run(KernelContext &ctx, const OutTensorCPU<Out, dims> &out,
           span<const InTensorCPU<In, dims>> in) {
    DALI_ENFORCE(in.size() == n_input_tensors_, make_string(
            "Input must have the same number of tensors as was specified in call to Setup. Expected: ",
            n_input_tensors_, "Actual: ", in.size()));
    for (int i = 0; i < n_input_tensors_; i++) {
      DALI_ENFORCE(in[i].shape == orig_shapes_[i], make_string(
              "Input must have the same shapes as was specified in call to Setup. Expected: ",
              orig_shapes_[i], "Actual: ", in[i].shape));
    }

    auto output = make_span(out);
    detail::TransferBuffers(output, output_shape_, in, axis_);


  }


  int axis_, n_input_tensors_;
  TensorShape<dims> output_shape_;
  std::vector<TensorShape<dims>> orig_shapes_;
};

}
}
#endif //DALI_TENSOR_STACK_CPU_H

