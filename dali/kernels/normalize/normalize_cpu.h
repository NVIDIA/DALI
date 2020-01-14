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
void normalize(Out *out, const In *in, int64_t count, const Param *mean, const Param *inv_stddev) {
  for (int64_t i = 0; i < count; i++) {
    out[i] = ConvertSat<Out>((in[i] - mean[i]) * inv_stddev[i]);
  }
}

template <typename Out, typename In, typename Param>
void normalize(Out *out, const In *in, int64_t count, const Param &mean, const Param &inv_stddev) {
  for (int64_t i = 0; i < count; i++) {
    out[i] = ConvertSat<Out>((in[i] - mean) * inv_stddev);
  }
}

template <typename Out, typename In, typename Param>
void normalize_inner(Out *out, const In *in, int64_t nouter, int64_t ninner,
                     const Param *mean, const Param *inv_stddev) {
  for (int64_t i = 0, k = 0; i < nouter; i++) {
    for (int64_t j = 0; j < ninner; j++, k++) {
      out[k] = ConvertSat<Out>((in[k] - mean[j]) * inv_stddev[j]);
    }
  }
}

template <typename Out, typename In, typename Param>
void normalize_outer(Out *out, const In *in, int64_t nouter, int64_t ninner,
                     const Param *mean, const Param *inv_stddev) {
  for (int64_t i = 0, k = 0; i < nouter; i++) {
    Param m = mean[i], d = inv_stddev[i];
    for (int64_t j = 0; j < ninner; j++, k++) {
      out[k] = ConvertSat<Out>((in[k] - m) * d);
    }
  }
}


}  // namespace normalize_impl

/**
 * @brief Subtracts mean and divides by standard deviation
 *
 * The kernel takes input tensor and produces equally shaped output tensor by subtracting mean
 * and multiplying by the inverse of standard deviation.
 * The result is converted (with rounding and saturation) to the specified output type.
 */
template <typename Out, typename In, typename MeanStdDev = float>
struct NormalizeCPU {
  KernelRequirements Setup(
    KernelContext &ctx,
    const InTensorCPU<In, -1> &in,
    const InTensorCPU<MeanStdDev, -1> &mean,
    const InTensorCPU<MeanStdDev, -1> &inv_stddev) {
      DALI_ENFORCE(mean.shape == inv_stddev.shape,
        "Mean and inverse standard deviation must have the same shape");
      DALI_ENFORCE(in.dim() == mean.dim() && in.dim() == inv_stddev.dim(),
        "All inputs must have equal dimensionality");
      int d = in.dim();
      for (int i = 0; i < d; i++) {
        DALI_ENFORCE(mean.shape[i] == 1 || mean.shape[i] == in.shape[i], make_string(
          "`mean` parameter's shape must have extent of 1 or one that matches the input.\n"
          "in.shape = ", in.shape,
          "\nmean.shape = ", mean.shape));
        DALI_ENFORCE(inv_stddev.shape[i] == 1 || inv_stddev.shape[i] == in.shape[i], make_string(
          "`inv_stddev` parameter's shape must have extent of 1 or one that matches the input.\n"
          "in.shape = ", in.shape,
          "\ninv_nstddev.shape = ", inv_stddev.shape));
      }
      this->input = in;
      this->orig_shape = in.shape;
      this->mean = mean;
      this->inv_stddev = inv_stddev;
      Squeeze();

      KernelRequirements req;
      ScratchpadEstimator se;
      req.output_shapes = { TensorListShape<>({in.shape}) };

      se.add<MeanStdDev>(AllocType::Host, inv_stddev.num_elements());

      req.scratch_sizes = se.sizes;
      return req;
  }

  void Run(KernelContext &ctx, const OutTensorCPU<Out, -1> &out) {
    (void)ctx;

    DALI_ENFORCE(out.shape == orig_shape, "Output and input shape must match");

    // squeeze the output
    this->output = make_tensor_cpu(out.data, input.shape);

    Normalize();
  }

  void Normalize() {
    int D = ndim();
    TensorShape<> data_strides = GetStrides(data_shape());
    TensorShape<> param_strides = GetStrides(param_shape());
    for (int i = 0; i < D; i++) {
      if (mean.shape[i] == 1)
        param_strides[i] = 0;  // reudced dim - use the same parameter slice for all data slices
    }
    NormalizeAxis(0, output.data, input.data, data_strides.data(),
                  mean.data, inv_stddev.data, param_strides.data());
  }

  void NormalizeAxis(int axis,
    Out *out, const In *in,
    const int64_t *data_strides,
    const MeanStdDev *mean, const MeanStdDev *stddev,
    const int64_t *param_strides) {

    using namespace normalize_impl;  // NOLINT

    if (axis == ndim() - 1) {
      assert(data_strides[axis] == 1);
      assert(param_strides[axis] <= 1);

      // last dimension - 1D case, which can be either:
      if (param_strides[axis] == 0)
        normalize(out, in, data_shape()[axis], *mean, *stddev);  // shared factors or..
      else
        normalize(out, in, data_shape()[axis], mean, stddev);    // per-element factors
    } else if (axis == ndim() - 2) {
      // 2D case can be either normalizing the innerm or the outer dimenion
      if (param_strides[axis] == 0 && param_strides[axis+1] != 0) {
        // e.g. normalize R,G,B interleaved channels using per-channel normalization factors
        // shared across pixels
        assert(param_strides[axis+1] == 1);
        normalize_inner(out, in, data_shape()[axis], data_shape()[axis+1], mean, stddev);
      } else {
        assert(param_strides[axis] != 0 && param_strides[axis+1] == 0);
        // e.g. normalize R,G,B planes using per-channel normalization factors shared across pixels
        assert(param_strides[axis] == 1);
        normalize_outer(out, in, data_shape()[axis], data_shape()[axis+1], mean, stddev);
      }
    } else {
      // anything else - just recursively peel off the outermost dimension
      ptrdiff_t data_ofs = 0, param_ofs = 0;
      ptrdiff_t data_stride = data_strides[axis];
      ptrdiff_t param_stride = param_strides[axis];
      for (int i = 0; i < data_shape()[axis];
           i++, data_ofs += data_stride, param_ofs += param_stride) {
        NormalizeAxis(axis + 1, out + data_ofs, in + data_ofs, data_strides,
                      mean + param_ofs, stddev + param_ofs, param_strides);
      }
    }
  }

  /**
   * @brief Collapses groups of reduced and groups of non-reduced dimensions.
   *
   * If the mean/stddev reduction spans multiple consecutive dimesnions, collapse them into one.
   * Conversely, if there are multiple non-reduced dimensions, also collapse them.
   * The function preserves boundaries between non-collapsed and collapsed dimensions, e.g.
   * If the input extent is 1, it can always be collapsed.
   *
   * in.shape  =  [ 4, 5, 7, 3, 2 ]
   * mean.shape = [ 4, 5, 1, 1, 2 ]
   *
   * will be collapsed to
   * in.shape  =  [ 20, 21, 2 ]
   * mean.shape = [ 20,  1, 2 ]
   *
   * The resulting simplified tensor is faster to traverse.
   */
  void Squeeze() {
    for (int i = 0; i < input.dim() - 1; i++) {
      bool collapse =
        (mean.shape[i] == 1 && mean.shape[i+1] == 1) ||
        (mean.shape[i] != 1 && mean.shape[i+1] != 1) ||
        input.shape[i] == 1 ||
        input.shape[i+1] == 1;

      if (collapse) {
        input      = collapse_dim(input, i);
        mean       = collapse_dim(mean, i);
        inv_stddev = collapse_dim(inv_stddev, i);
        i--;
      }
    }
  }

  inline int ndim() const noexcept { return input.dim(); }
  inline const TensorShape<> &data_shape() const noexcept  { return input.shape; }
  inline const TensorShape<> &param_shape() const noexcept { return mean.shape; }

  InTensorCPU<In, -1> input;
  OutTensorCPU<Out, -1> output;
  TensorShape<> orig_shape;
  InTensorCPU<MeanStdDev, -1> mean, inv_stddev;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_NORMALIZE_NORMALIZE_CPU_H_
