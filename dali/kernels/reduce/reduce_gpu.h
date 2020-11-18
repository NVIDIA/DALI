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

#ifndef DALI_KERNELS_REDUCE_REDUCE_GPU_H_
#define DALI_KERNELS_REDUCE_REDUCE_GPU_H_

/**
 * @file
 *
 * This file contains classes for performing various reductions on GPU.
 */

#include <memory>
#include "dali/kernels/kernel.h"
#include "dali/core/host_dev.h"
#include "dali/core/tensor_view.h"


namespace dali {
namespace kernels {

/**
 * @brief Calculates the sum of elements in the tensor(s) along given axes
 *
 * @details
 * The reduction can be described by the following algorithm:
 *
 * ```
 * function reduce(out, in, dim, in_coords, out_coords, in_size, axes)
 *   if dim == number_of_dimensions(in) - 1
 *     if in_size[dim] == 0:
 *       out[out_coords] = neutral_element
 *     else
 *       out[out_coords] = reduce(out[out_coords], in[in_coords])
 *   else
 *     for i = 0 to in_size[dim]
 *       if axes contains dim
 *         out_coords[dim] = 0
 *       else
 *         out_coords[dim] = i
 *       reduce(out, in, dim + 1, in_coords, out_coords, in_size, axes)
 * ```
 *
 * If batch reduction is requested, the corresponding elements of the output tensors
 * (calculated as in the algorithm above) are also reduced and the output batch contains
 * just one tensor.
 *
 * For batch reduction to be possible, non-reduced extents of all samples must be equal.
 */
template <typename Out, typename In>
class DLL_PUBLIC SumGPU {
 public:
  SumGPU();
  ~SumGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class SumGPU<uint64_t, uint8_t>;
extern template class SumGPU<float, uint8_t>;
extern template class SumGPU<int64_t, int8_t>;
extern template class SumGPU<float, int8_t>;

extern template class SumGPU<uint64_t, uint16_t>;
extern template class SumGPU<float, uint16_t>;
extern template class SumGPU<int64_t, int16_t>;
extern template class SumGPU<float, int16_t>;

extern template class SumGPU<uint64_t, uint32_t>;
extern template class SumGPU<float, uint32_t>;
extern template class SumGPU<int64_t, int32_t>;
extern template class SumGPU<float, int32_t>;

extern template class SumGPU<uint8_t, uint8_t>;
extern template class SumGPU<int8_t, int8_t>;
extern template class SumGPU<uint16_t, uint16_t>;
extern template class SumGPU<int16_t, int16_t>;
extern template class SumGPU<uint32_t, uint32_t>;
extern template class SumGPU<int32_t, int32_t>;
extern template class SumGPU<uint64_t, uint64_t>;
extern template class SumGPU<int64_t, int64_t>;
extern template class SumGPU<float, float>;


/**
 * @brief Calculates the min of elements in the tensor(s) along given axes
 *
 * @details
 * The reduction can be described by the following algorithm:
 *
 * ```
 * function reduce(out, in, dim, in_coords, out_coords, in_size, axes)
 *   if dim == number_of_dimensions(in) - 1
 *     if in_size[dim] == 0:
 *       out[out_coords] = neutral_element
 *     else
 *       out[out_coords] = reduce(out[out_coords], in[in_coords])
 *   else
 *     for i = 0 to in_size[dim]
 *       if axes contains dim
 *         out_coords[dim] = 0
 *       else
 *         out_coords[dim] = i
 *       reduce(out, in, dim + 1, in_coords, out_coords, in_size, axes)
 * ```
 *
 * If batch reduction is requested, the corresponding elements of the output tensors
 * (calculated as in the algorithm above) are also reduced and the output batch contains
 * just one tensor.
 *
 * For batch reduction to be possible, non-reduced extents of all samples must be equal.
 */
template <typename Out, typename In>
class DLL_PUBLIC MinGPU {
 public:
  MinGPU();
  ~MinGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class MinGPU<uint8_t, uint8_t>;
extern template class MinGPU<int8_t, int8_t>;
extern template class MinGPU<uint16_t, uint16_t>;
extern template class MinGPU<int16_t, int16_t>;
extern template class MinGPU<uint32_t, uint32_t>;
extern template class MinGPU<int32_t, int32_t>;
extern template class MinGPU<uint64_t, uint64_t>;
extern template class MinGPU<int64_t, int64_t>;
extern template class MinGPU<float, float>;

/**
 * @brief Calculates the max of elements in the tensor(s) along given axes
 *
 * @details
 * The reduction can be described by the following algorithm:
 *
 * ```
 * function reduce(out, in, dim, in_coords, out_coords, in_size, axes)
 *   if dim == number_of_dimensions(in) - 1
 *     if in_size[dim] == 0:
 *       out[out_coords] = neutral_element
 *     else
 *       out[out_coords] = reduce(out[out_coords], in[in_coords])
 *   else
 *     for i = 0 to in_size[dim]
 *       if axes contains dim
 *         out_coords[dim] = 0
 *       else
 *         out_coords[dim] = i
 *       reduce(out, in, dim + 1, in_coords, out_coords, in_size, axes)
 * ```
 *
 * If batch reduction is requested, the corresponding elements of the output tensors
 * (calculated as in the algorithm above) are also reduced and the output batch contains
 * just one tensor.
 *
 * For batch reduction to be possible, non-reduced extents of all samples must be equal.
 */
template <typename Out, typename In>
class DLL_PUBLIC MaxGPU {
 public:
  MaxGPU();
  ~MaxGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class MaxGPU<uint8_t, uint8_t>;
extern template class MaxGPU<int8_t, int8_t>;
extern template class MaxGPU<uint16_t, uint16_t>;
extern template class MaxGPU<int16_t, int16_t>;
extern template class MaxGPU<uint32_t, uint32_t>;
extern template class MaxGPU<int32_t, int32_t>;
extern template class MaxGPU<uint64_t, uint64_t>;
extern template class MaxGPU<int64_t, int64_t>;
extern template class MaxGPU<float, float>;

/**
 * @brief Calculates the mean of elements in the tensor(s) along given axes
 *
 * @copydetails SumGPU
 *
 * Output elements are calculated as the sum of input elements divided by the reduction factor,
 * given as:
 * per-sample reduction:
 * ```
 * reduction_factor[sample] = product(in_shape[sample][i] for i in axes)
 * ```
 * batch reduction:
 * ```
 * reduction_factor =
 * sum(
 *   product(in_shape[sample][i] for i in axes)
 *   for sample from 0 to number_of_samples
 * )
 * ```
 *
 * The reduction factor cannot be zero - if it is, an exception is thrown.
 */
template <typename Out, typename In>
class DLL_PUBLIC MeanGPU {
 public:
  MeanGPU();
  ~MeanGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};


extern template class MeanGPU<uint8_t, uint8_t>;
extern template class MeanGPU<float, uint8_t>;
extern template class MeanGPU<int8_t, int8_t>;
extern template class MeanGPU<float, int8_t>;

extern template class MeanGPU<uint16_t, uint16_t>;
extern template class MeanGPU<float, uint16_t>;
extern template class MeanGPU<int16_t, int16_t>;
extern template class MeanGPU<float, int16_t>;

extern template class MeanGPU<uint32_t, uint32_t>;
extern template class MeanGPU<float, uint32_t>;
extern template class MeanGPU<int32_t, int32_t>;
extern template class MeanGPU<float, int32_t>;

extern template class MeanGPU<int64_t, int64_t>;
extern template class MeanGPU<float, int64_t>;
extern template class MeanGPU<uint64_t, uint64_t>;
extern template class MeanGPU<float, uint64_t>;

extern template class MeanGPU<float, float>;


/**
 * @brief Calculates the mean square of elements in the tensor(s) along given axes
 *
 * Output elements are calculated as a mean of squared input elements.
 * See MeanGPU for details on calculating the mean.
 */
template <typename Out, typename In>
class DLL_PUBLIC MeanSquareGPU {
 public:
  MeanSquareGPU();
  ~MeanSquareGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class MeanSquareGPU<uint64_t, uint8_t>;
extern template class MeanSquareGPU<float, uint8_t>;
extern template class MeanSquareGPU<int64_t, int8_t>;
extern template class MeanSquareGPU<float, int8_t>;

extern template class MeanSquareGPU<uint64_t, uint16_t>;
extern template class MeanSquareGPU<float, uint16_t>;
extern template class MeanSquareGPU<int64_t, int16_t>;
extern template class MeanSquareGPU<float, int16_t>;

extern template class MeanSquareGPU<uint64_t, uint32_t>;
extern template class MeanSquareGPU<float, uint32_t>;
extern template class MeanSquareGPU<int64_t, int32_t>;
extern template class MeanSquareGPU<float, int32_t>;

extern template class MeanSquareGPU<uint64_t, uint64_t>;
extern template class MeanSquareGPU<float, uint64_t>;
extern template class MeanSquareGPU<int64_t, int64_t>;
extern template class MeanSquareGPU<float, int64_t>;

extern template class MeanSquareGPU<float, float>;

/**
 * @brief Calculates the root mean square of elements in the tensor(s) along given axes
 *
 * Output elements are calculated as a squre root of the mean of squared input elements.
 * See MeanGPU for details on calculating the mean.
 */
template <typename Out, typename In>
class DLL_PUBLIC RootMeanSquareGPU {
 public:
  RootMeanSquareGPU();
  ~RootMeanSquareGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out, const InListGPU<In> &in);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class RootMeanSquareGPU<uint8_t, uint8_t>;
extern template class RootMeanSquareGPU<float, uint8_t>;
extern template class RootMeanSquareGPU<int8_t, int8_t>;
extern template class RootMeanSquareGPU<float, int8_t>;

extern template class RootMeanSquareGPU<uint16_t, uint16_t>;
extern template class RootMeanSquareGPU<float, uint16_t>;
extern template class RootMeanSquareGPU<int16_t, int16_t>;
extern template class RootMeanSquareGPU<float, int16_t>;

extern template class RootMeanSquareGPU<uint32_t, uint32_t>;
extern template class RootMeanSquareGPU<float, uint32_t>;
extern template class RootMeanSquareGPU<int32_t, int32_t>;
extern template class RootMeanSquareGPU<float, int32_t>;

extern template class RootMeanSquareGPU<uint64_t, uint64_t>;
extern template class RootMeanSquareGPU<float, uint64_t>;
extern template class RootMeanSquareGPU<int64_t, int64_t>;
extern template class RootMeanSquareGPU<float, int64_t>;

extern template class RootMeanSquareGPU<float, float>;


/**
 * @brief Calculates the standard deviation of input elements along given axes, given externally
 *        provided mean values.
 *
 * True standard deviation would be calculated as:
 * ```
 * sqrt(mean( (in - mean(in)) ^ 2 )
 * ```
 * Here, the `mean(in)` term is not calculated internally, but externally provided as a tensor.
 *
 * For more details on how directional reductions work, see SumGPU, MeanGPU, RootMeanSquareGPU
 */
template <typename Out, typename In, typename Mean = Out>
class DLL_PUBLIC StdDevGPU {
 public:
  StdDevGPU();
  ~StdDevGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   *
   * @param ddof delta degrees of freedom for Bessel's correction
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out,
           const InListGPU<In> &in, const InListGPU<Mean> &mean, int ddof = 0);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class StdDevGPU<uint8_t, uint8_t>;
extern template class StdDevGPU<float, uint8_t>;
extern template class StdDevGPU<int8_t, int8_t>;
extern template class StdDevGPU<float, int8_t>;

extern template class StdDevGPU<uint16_t, uint16_t>;
extern template class StdDevGPU<float, uint16_t>;
extern template class StdDevGPU<int16_t, int16_t>;
extern template class StdDevGPU<float, int16_t>;

extern template class StdDevGPU<uint32_t, uint32_t>;
extern template class StdDevGPU<float, uint32_t>;
extern template class StdDevGPU<int32_t, int32_t>;
extern template class StdDevGPU<float, int32_t>;

extern template class StdDevGPU<uint64_t, uint64_t>;
extern template class StdDevGPU<float, uint64_t>;
extern template class StdDevGPU<int64_t, int64_t>;
extern template class StdDevGPU<float, int64_t>;

extern template class StdDevGPU<float, float>;


/**
 * @brief Calculates the variance of input elements along given axes, given externally
 *        provided mean values.
 *
 * Here, the `mean(in)` term is not calculated internally, but externally provided as a tensor.
 *
 * For more details on how directional reductions work, see SumGPU, MeanGPU, RootMeanSquareGPU
 */
template <typename Out, typename In, typename Mean = Out>
class DLL_PUBLIC VarianceGPU {
 public:
  VarianceGPU();
  ~VarianceGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  /**
   * @brief Performs the reduction, according to the parameters specified in Setup.
   *
   * @param ddof delta degrees of freedom for Bessel's correction
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out,
           const InListGPU<In> &in, const InListGPU<Mean> &mean, int ddof = 0);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class VarianceGPU<uint8_t, uint8_t>;
extern template class VarianceGPU<float, uint8_t>;
extern template class VarianceGPU<int8_t, int8_t>;
extern template class VarianceGPU<float, int8_t>;

extern template class VarianceGPU<uint16_t, uint16_t>;
extern template class VarianceGPU<float, uint16_t>;
extern template class VarianceGPU<int16_t, int16_t>;
extern template class VarianceGPU<float, int16_t>;

extern template class VarianceGPU<uint32_t, uint32_t>;
extern template class VarianceGPU<float, uint32_t>;
extern template class VarianceGPU<int32_t, int32_t>;
extern template class VarianceGPU<float, int32_t>;

extern template class VarianceGPU<uint64_t, uint64_t>;
extern template class VarianceGPU<float, uint64_t>;
extern template class VarianceGPU<int64_t, int64_t>;
extern template class VarianceGPU<float, int64_t>;

extern template class VarianceGPU<float, float>;


/**
 * @brief Calculates the inverse of standard deviation of input elements along given axes,
 *        given externally provided mean values.
 *
 * The output values are calculated as:
 * ```
 * s = sum( (in[pos] - mean[reduced_pos])^2 )
 * out[reduced_pos] = s > 0 || reg > 0
 *                    ? 1/sqrt(s / reduction_factor + reg)
 *                    : 0
 * ```
 * where `reduction_factor` is the number of input elements contributing to a single output.
 *
 * For more details on how directional reductions work, see SumGPU, MeanGPU, RootMeanSquareGPU.
 *
 * @see StdDevGPU
 */
template <typename Out, typename In, typename Mean = Out>
class DLL_PUBLIC InvStdDevGPU {
 public:
  InvStdDevGPU();
  ~InvStdDevGPU();

  /**
   * @brief Sets up the reduction
   *
   * Sets up the reduction according to the parameters. The indices of dimensions to be reduced
   * are provided in `axes` parameter.
   * For a successful batch reduction, the reduced shape of all samples must be equal (but the
   * input may have non-uniform shape, as long as the non-uniform dimensions are reduced).
   *
   * @param ctx          the execution environment
   * @param in_shape     shape of the input tensor list
   * @param axes         indices of axes to reduce along
   * @param keep_dims    if true, the reduced dimensions are kept in the output shape, with the
   *                     extent of 1
   * @param reduce_batch if true, reduces respective output values of all samples in the batch
   *                     and outputs a single tensor
   */
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &in_shape,
                           span<const int> axes, bool keep_dims, bool reduce_batch);

  using param_t = std::conditional_t<std::is_same<Out, double>::value, double, float>;

  /**
   * @brief Calculates regularized inverse standard deviation
   *
   * The output values are calculated as:
   * ```
   * s = sum( (in[pos] - mean[reduced_pos])^2 )
   * out[reduced_pos] = (s > 0 || reg > 0) && reduction_factor - ddof > 0
   *                    ? 1/sqrt(s / (reduction_factor - ddof) + reg)
   *                    : 0
   * ```
   * where `reduction_factor` is the number of input elements contributing to a single output.
   *
   * @param ctx     the execution environment
   * @param out     (regularized) inverse standard deviation
   * @param in      input tensor
   * @param mean    mean, used for centering the data
   * @param ddof    delta degrees of freedom, for Bessel's correction
   * @param reg     regularizing term to avoid division by zero (or small numbers);
   *                it's added to the sum of squares in variance calculation, preventing
   *                it from being close to zero; if reg = 0, the results that would
   *                cause division by zero are forced to 0, but small denominators
   *                may still cause problems
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out,
           const InListGPU<In> &in, const InListGPU<Mean> &mean, int ddof = 0, param_t epsilon = 0);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

extern template class InvStdDevGPU<float, uint8_t>;
extern template class InvStdDevGPU<float, int8_t>;

extern template class InvStdDevGPU<float, uint16_t>;
extern template class InvStdDevGPU<float, int16_t>;

extern template class InvStdDevGPU<float, uint32_t>;
extern template class InvStdDevGPU<float, int32_t>;

extern template class InvStdDevGPU<float, float>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_REDUCE_GPU_H_
