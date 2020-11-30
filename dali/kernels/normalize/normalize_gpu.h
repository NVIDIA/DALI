// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_H_
#define DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_H_

/**
 * @file
 *
 * This file contains the GPU kernel for directional data normalization.
 */

#include <memory>
#include "dali/core/host_dev.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

/**
 * @brief Normalizes data
 *
 * Data normalization is done using externally provided base (typically: mean or min) and scale
 * (typically reciprocal of standard deviation or 1/(max-min)).
 * The normalization follows the formula:
 * ```
 * out[data_idx] = (in[data_idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
 * ```
 * Where `data_idx` is a position in the data tensor (in, out) and `param_idx` is a position
 * in the base and scale tensors (see below for details). The two additional constants,
 * `global_scale` and `shift` can be used to adjust the result to the dynamic range and resolution
 * of the output type.
 *
 * The `scale` parameter may also be interpreted as standard deviation - in that case, its
 * reciprocal is used and optionally, a regularizing term is added to the variance.
 * ```
 * m = 1 / sqrt(square(stddev[param_idx]) + epsilon)
 * out[data_idx] = (in[data_idx] - mean[param_idx]) * m * global_scale + shift
 * ```
 *
 * The shapes of the input/output data and of the parameters (base/scale) can be different - some
 * dimensions in the parameter tensors may have extent 1, in which case the values are broadcast
 * along this axis, as they would in numpy when operating on arrays of different shape.
 * `param_idx` is calculated as follows:
 * ```
 * param_idx[axis] = param_shape[axis] == 1 ? 0 : data_idx[axis]
 * ```
 *
 * The parameter batch can be of the same length as the data batch or have length == 1 - in which
 * case the first sample is used for normalization of all input tensors.
 *
 * One or both of the parameters can be scalars.
 *
 * @tparam In   type of elements in the input tensor
 * @tparam Out  type of elements in the output tensor; if it's an integral type, the results
 *              are rounded to the nearest integer and clamped to avoid overflow
 *
 */
template <typename Out, typename In>
class DLL_PUBLIC NormalizeGPU {
 public:
  NormalizeGPU();
  ~NormalizeGPU();

  using Base = float;
  using Scale = float;

  /**
   * @brief Sets up the normalization
   *
   * @param data_shape      shape of the input and output tensor lists
   * @param param_shape     shape of the parameters (base, scale)
   *                        it must have the same number of dimsensions as data_shape and the
   *                        extents must be either equal to respective extents in data_shape or 1;
   *                        it can have either the same number of samples as data_shape
   *                        or one sample;
   *                        if both scalar_base and scalar_scale are true, param_shape is ignored
   * @param scalar_base     if true, the Run overload with scalar `base` parameter must be used
   * @param scalar_scale    if true, the Run overload with scalar `scale` parameter must be used
   * @param scale_is_stddev if true, scale is interpreted as standard deviation and it's regularized
   *                        and its reciprocal is used when scaling
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &data_shape,
                           const TensorListShape<> &param_shape,
                           bool scalar_base,
                           bool scalar_scale,
                           bool scale_is_stddev);

  /**
   * @brief Sets up the normalization
   *
   * @param data_shape      shape of the input and output tensor lists
   * @param axes            indices of axes along which the parameters are broadcast;
   *                        ignored if both scalar_base and scalar_scale are `true`
   * @param scalar_base     if true, the Run overload with scalar `base` parameter must be used
   * @param scalar_scale    if true, the Run overload with scalar `scale` parameter must be used
   * @param scale_is_stddev if true, scale is interpreted as standard deviation and it's regularized
   *                        and its reciprocal is used when scaling
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &data_shape,
                           span<const int> axes,
                           bool scalar_base,
                           bool scalar_scale,
                           bool scale_is_stddev);

  ///@{
  /**
   * @brief Normalizes the data with base and scale given as tensor lists
   *
   * @param ctx   kernel's execution context - scratchpad and CUDA stream
   * @param out   output tensor list, must match data_shape
   * @param in    input tensor list, must match data_shape
   * @param base  value(s) subtracted from input elements; must match param_shape
   * @param scale value(s) of values of scales (or standard deviations), must match param_shape
   * @param global_scale  additional scaling factor, used e.g. when output is of integral type
   * @param shift         additional bias value, used e.g. when output is of unsigned type
   * @param epsilon       regularizing term added to variance; only used if scale_is_stddev = true
   *                      was specified in Setup
   */
  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           const InListGPU<Base> &base, const InListGPU<Scale> &scale,
           float global_scale = 1, float shift = 0, float epsilon = 0);

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           float base, const InListGPU<Scale> &scale,
           float global_scale = 1, float shift = 0, float epsilon = 0);

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           const InListGPU<Base> &base, float scale,
           float global_scale = 1, float shift = 0, float epsilon = 0);

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           float base, float scale,
           float global_scale = 1, float shift = 0, float epsilon = 0);
  ///@}

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#define DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, output_type)\
  linkage template class NormalizeGPU<output_type, int8_t>;\
  linkage template class NormalizeGPU<output_type, int16_t>;\
  linkage template class NormalizeGPU<output_type, int32_t>;\
  linkage template class NormalizeGPU<output_type, uint8_t>;\
  linkage template class NormalizeGPU<output_type, uint16_t>;\
  linkage template class NormalizeGPU<output_type, uint32_t>;\
  linkage template class NormalizeGPU<output_type, float>;\

#define DALI_INSTANTIATE_NORMALIZE_GPU(linkage)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, int8_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, int16_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, int32_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, uint8_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, uint16_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, uint32_t)\
  DALI_INSTANTIATE_NORMALIZE_GPU_OUT(linkage, float)

DALI_INSTANTIATE_NORMALIZE_GPU(extern)

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_H_
