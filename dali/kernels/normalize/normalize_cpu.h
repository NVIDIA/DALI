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

#ifndef DALI_KERNELS_NORMALIZE_NORMALIZE_CPU_H_
#define DALI_KERNELS_NORMALIZE_NORMALIZE_CPU_H_

#include <cassert>
#include <utility>
#include <vector>
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_view.h"
#include "dali/core/util.h"
#include "dali/core/convert.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace kernels {

namespace normalize_impl {

template <typename Out, typename In, typename Param>
void normalize(Out *out, const In *in, int64_t count,
               const Param *mean, const Param *scale, Param shift) {
  #pragma omp simd
  for (int64_t i = 0; i < count; i++) {
    out[i] = ConvertSat<Out>((in[i] - mean[i]) * scale[i] + shift);
  }
}

template <typename Out, typename In, typename Param>
void normalize(Out *out, const In *in, int64_t count,
               Param mean, Param scale, Param shift) {
  #pragma omp simd
  for (int64_t i = 0; i < count; i++) {
    out[i] = ConvertSat<Out>((in[i] - mean) * scale + shift);
  }
}

template <typename Out, typename In, typename Param>
void normalize_inner(Out *out, const In *in, int64_t nouter, int64_t ninner,
                     const Param *mean, const Param *scale, Param shift) {
  for (int64_t i = 0, k = 0; i < nouter; i++, k += ninner) {
    #pragma omp simd
    for (int64_t j = 0; j < ninner; j++) {
      out[k + j] = ConvertSat<Out>((in[k + j] - mean[j]) * scale[j] + shift);
    }
  }
}

template <typename Out, typename In, typename Param>
void normalize_outer(Out *out, const In *in, int64_t nouter, int64_t ninner,
                     const Param *mean, const Param *scale, Param shift) {
  for (int64_t i = 0, k = 0; i < nouter; i++, k += ninner) {
    Param m = mean[i], d = scale[i];
    #pragma omp simd
    for (int64_t j = 0; j < ninner; j++) {
      out[k + j] = ConvertSat<Out>((in[k + j] - m) * d + shift);
    }
  }
}


}  // namespace normalize_impl

/**
 * @brief Subtracts mean and divides by standard deviation
 *
 * The kernel takes input tensor and produces equally shaped output tensor by subtracting mean
 * and multiplying by a scaling factor (and optionally adding a scalar value to all outputs)
 * The result is converted (with rounding and saturation) to the specified output type.
 */
template <typename Out, typename In, typename Param = float>
struct NormalizeCPU {
  KernelRequirements Setup(
        KernelContext &ctx,
        const TensorShape<> &in_shape,
        const TensorShape<> &param_shape) {
    int d = in_shape.size();
    for (int i = 0; i < d; i++) {
      DALI_ENFORCE(param_shape[i] == 1 || param_shape[i] == in_shape[i], make_string(
        "Normalization parameters' shape must have extent of 1 or one that matches the input.\n"
        "in_shape = ", in_shape,
        "\nparam_shape = ", param_shape));
    }

    orig_shape_ = in_shape;
    data_shape_ = in_shape;
    param_shape_ = param_shape;
    orig_param_shape_ = param_shape;
    Squeeze();

    KernelRequirements req;
    req.output_shapes.resize(1);
    TensorListShape<> tmp({in_shape});  // clang's destructor bug still haunting
    req.output_shapes[0] = tmp;
    return req;
  }

  /**
   * @param out   output tensor, must have the same shape as the input
   * @param in    input tensor, must have the same shape as specified in Setup
   * @param mean  mean values to be subtracted from input data
   * @param scale scale (typically inverse of standard deviation) to multiply the centered values
   * @param shift the value to add to centered and rescaled data; input values equal to mean
   *              will produce shift (converted to Out) at the output
   */
  void Run(KernelContext &ctx,
           const OutTensorCPU<Out, -1> &out,
           const InTensorCPU<In, -1> &in,
           const InTensorCPU<Param, -1> &mean,
           const InTensorCPU<Param, -1> &scale,
           Param shift = 0) {
    (void)ctx;

    DALI_ENFORCE(mean.shape == scale.shape, make_string(
        "Mean and scale must have the same shape; got:"
        "\nmean.shape       = ", mean.shape,
        "\nscale.shape = ", scale.shape));

    DALI_ENFORCE(in.shape == orig_shape_,
      "Input must have the same shape as was specified in call to Setup");

    DALI_ENFORCE(out.shape == orig_shape_, "Output and input shapes must match");

    input_ = in.data;
    output_ = out.data;

    mean_ = mean.data;
    scale_ = scale.data;

    shift_ = shift;

    Normalize();
  }

  void Normalize() {
    int D = ndim();
    TensorShape<> data_strides = GetStrides(data_shape_);
    TensorShape<> param_strides = GetStrides(param_shape_);
    for (int i = 0; i < D; i++) {
      if (param_shape_[i] == 1)
        param_strides[i] = 0;  // reduced dim - use the same parameter slice for all data slices
    }
    NormalizeAxis(0, output_, input_, data_strides.data(),
                  mean_, scale_, param_strides.data());
  }

  void NormalizeAxis(int axis,
    Out *out, const In *in,
    const int64_t *data_strides,
    const Param *mean, const Param *scale,
    const int64_t *param_strides) {

    using namespace normalize_impl;  // NOLINT

    if (axis == ndim() - 1) {
      assert(data_strides[axis] == 1);
      assert(param_strides[axis] <= 1);

      // last dimension - 1D case, which can be either:
      if (param_strides[axis] == 0)
        normalize(out, in, data_shape_[axis], *mean, *scale, shift_);  // shared factors or..
      else
        normalize(out, in, data_shape_[axis], mean, scale, shift_);    // per-element factors
    } else if (axis == ndim() - 2) {
      // 2D case can be either normalizing the inner or the outer dimension
      if (param_strides[axis] == 0 && param_strides[axis+1] != 0) {
        // e.g. normalize R,G,B interleaved channels using per-channel normalization factors
        // shared across pixels
        assert(param_strides[axis+1] == 1);
        normalize_inner(out, in, data_shape_[axis], data_shape_[axis+1], mean, scale, shift_);
      } else {
        assert(param_strides[axis] == 1 && param_strides[axis+1] == 0);
        // e.g. normalize R,G,B planes using per-channel normalization factors shared across pixels
        normalize_outer(out, in, data_shape_[axis], data_shape_[axis+1], mean, scale, shift_);
      }
    } else {
      // anything else - just recursively peel off the outermost dimension
      ptrdiff_t data_ofs = 0, param_ofs = 0;
      ptrdiff_t data_stride = data_strides[axis];
      ptrdiff_t param_stride = param_strides[axis];
      for (int i = 0; i < data_shape_[axis];
           i++, data_ofs += data_stride, param_ofs += param_stride) {
        NormalizeAxis(axis + 1, out + data_ofs, in + data_ofs, data_strides,
                      mean + param_ofs, scale + param_ofs, param_strides);
      }
    }
  }

  /**
   * @brief Collapses groups of reduced and groups of non-reduced dimensions.
   *
   * If the mean/scale reduction spans multiple consecutive dimensions, collapse them into one.
   * Conversely, if there are multiple non-reduced dimensions, also collapse them.
   * The function preserves boundaries between non-collapsed and collapsed dimensions, e.g.
   * If an extent is 1, it can always be - it's compatible with both reduced and non-reduced
   * dimensions.
   *
   * data_shape   = [ 4, 5, 7, 3, 2 ]
   * param_shape_ = [ 4, 5, 1, 1, 2 ]
   *
   * will be collapsed to
   * data_shape_  = [ 20, 21, 2 ]
   * param_shape_ = [ 20,  1, 2 ]
   *
   * The resulting simplified tensor is faster to traverse.
   */
  void Squeeze() {
    for (int i = 0; i < ndim() - 1; i++) {
      bool collapse =
        (param_shape_[i] == 1 && param_shape_[i+1] == 1) ||
        (param_shape_[i] != 1 && param_shape_[i+1] != 1) ||
        data_shape_[i] == 1 ||
        data_shape_[i+1] == 1;

      if (collapse) {
        data_shape_ = collapse_dim(data_shape_, i);
        param_shape_ = collapse_dim(param_shape_, i);
        i--;
      }
    }
  }

  inline int ndim() const noexcept { return data_shape_.size(); }

  TensorShape<> orig_shape_, orig_param_shape_, data_shape_, param_shape_;
  const In *input_  = nullptr;;
  Out *output_ = nullptr;
  const Param *mean_, *scale_;
  Param shift_ = 0;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_NORMALIZE_NORMALIZE_CPU_H_
