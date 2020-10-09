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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_CPU_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_CPU_H_

#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/join/tensor_join_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/format.h"

namespace dali {
namespace kernels {
namespace tensor_join {

template<typename T>
void
ConcatenateTensors(TensorView<StorageCPU, T> output,
                   span<const TensorView<StorageCPU, const T>> inputs,
                   int axis) {
  SmallVector<int64_t, 64> copy_sizes;
  copy_sizes.resize(inputs.size());
  for (int t = 0; t < inputs.size(); t++) {
    copy_sizes[t] = volume(inputs[t].shape.begin() + axis, inputs[t].shape.end());
  }
  auto nouter = volume(output.shape.begin(), output.shape.begin() + axis);
  auto *out = output.data;
  for (ptrdiff_t outer = 0; outer < nouter; outer++) {
    for (int t = 0; t < inputs.size(); t++) {
      auto *src = inputs[t].data + outer * copy_sizes[t];
      for (ptrdiff_t inner = 0; inner < copy_sizes[t]; inner++) {
        *out++ = src[inner];
      }
    }
  }
}

}  // namespace tensor_join

/**
 * Joins multiple input tensors into one output tensor, along given axis.
 *
 * The kernel works in 2 modes: STACK and CONCAT. In CONCAT mode, kernel creates a new tensor,
 * with joined values along given dimension, e.g.
 *
 * arr0 = [[1, 2, 4, 2], [1, 1, 7, 6], [6, 8, 8, 4]]
 * shape = (3, 4)
 *
 * arr1 = [[3, 8, 8, 6], [8, 1, 5, 7], [6, 2, 7, 5]]
 * shape = (3, 4)
 *
 * concatenate([arr0, arr1], axis=1) -> [[1, 2, 4, 2, 3, 8, 8, 6],
 *                                       [1, 1, 7, 6, 8, 1, 5, 7],
 *                                       [6, 8, 8, 4, 6, 2, 7, 5]]
 * shape = (3, 8)
 *
 * Stacking, on the other hand, creates new tensor with added dimension, where `axis` points.
 *
 * stack([arr0, arr1], axis=1) -> [[[1, 2, 4, 2],
 *                                  [3, 8, 8, 6]],
 *
 *                                 [[1, 1, 7, 6],
 *                                  [8, 1, 5, 7]],
 *
 *                                 [[6, 8, 8, 4],
 *                                  [6, 2, 7, 5]]]
 * shape = (3, 2, 4)
 *
 * @tparam new_axis if true, STACK mode is applied.
 */
template<typename T, bool new_axis = true, int dims = -1>
struct TensorJoinCPU {
  ///@{
  /**
   * @param in_shapes Shapes of input tensors.
   * @param axis Axis, along which tensors will be joined.
   */
  KernelRequirements Setup(KernelContext &ctx, span<const TensorShape<dims>> in_shapes, int axis) {
    n_input_tensors_ = in_shapes.size();
    auto ndims = in_shapes[0].sample_dim();
    DALI_ENFORCE(axis >= 0 && axis <= ndims - !new_axis,
                 make_string("Incorrect axis. Actual: ", axis, ". Expected in [0, ",
                             ndims - !new_axis, "] range (", new_axis ? "STACK" : "CONCAT",
                             " mode)"));
    axis_ = axis;

    {
      const auto &ref = in_shapes[0];
      for (int i = 1; i < n_input_tensors_; i++) {
        DALI_ENFORCE(in_shapes[i].sample_dim() == ref.sample_dim(),
                     "Every input shape must have the same number of dimensions.");
        for (int j = 0; j < ref.size(); j++) {
          if (!new_axis) {
            DALI_ENFORCE(in_shapes[i][j] == ref.shape[j] || j == axis_, make_string(
                    "Number of samples in every dimension (except the concatenated one) "
                    "must be the same (CONCAT mode). 0-th shape at index ", j, " has dimension ",
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
    output_shape_ = tensor_join::JoinedShape(in_shapes, axis_, new_axis);
    kr.output_shapes.resize(1);
    TensorListShape<> tmp({output_shape_});  // clang's destructor bug still haunting
    kr.output_shapes[0] = tmp;
    return kr;
  }


  KernelRequirements Setup(KernelContext &ctx, span<const InTensorCPU<T, dims>> in, int axis) {
    std::vector<TensorShape<>> in_shapes(in.size());
    for (int i = 0; i < in.size(); i++) {
      in_shapes[i] = in[i].shape;
    }
    return Setup(ctx, make_span(in_shapes), axis);
  }
  ///@}

  static constexpr int output_dims = (dims == DynamicDimensions ? DynamicDimensions :
                                      (new_axis ? dims + 1 : dims));


  /**
   * @param out output tensor. Must be properly allocated
   * @param in input tensors. The number of these tensors, as well as their shapes,
   *           must match with what is provided in Setup call.
   */
  void Run(KernelContext &ctx, OutTensorCPU<T, output_dims> out,
           span<const InTensorCPU<T>> in) {
    if (in.size() != n_input_tensors_) {
      throw std::invalid_argument(make_string(
              "Input must have the same number of tensors as was specified in call to Setup.\n"
              "Expected: ", n_input_tensors_, "\nActual: ", in.size()));
    }
    if (out.shape != output_shape_) {
      throw std::invalid_argument(
              make_string("Output has incorrect shape.\nExpected: ", output_shape_, "\nActual: ",
                          out.shape));
    }

    tensor_join::ConcatenateTensors(out, in, axis_);
  }


  int axis_ = -1, n_input_tensors_ = -1;
  TensorShape<dims> output_shape_;
};

///@{
/**
 * @see TensorJoinCPU
 */
template<typename T, int ndims = -1>
using TensorStackCPU = TensorJoinCPU<T, true, ndims>;

template<typename T, int ndims = -1>
using TensorConcatCPU = TensorJoinCPU<T, false, ndims>;
///@}

}  // namespace kernels
}  // namespace dali
#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_CPU_H_
