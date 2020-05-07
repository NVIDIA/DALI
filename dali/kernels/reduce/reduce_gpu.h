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

#include <memory>
#include "dali/kernels/kernel.h"
#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {

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

extern template class SumGPU<float, float>;


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

extern template class MeanGPU<float, float>;


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

extern template class RootMeanSquareGPU<float, float>;


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
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out,
           const InListGPU<In> &in, const InListGPU<Mean> &mean);

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

extern template class StdDevGPU<float, float>;


template <typename Out, typename In, typename Mean = In>
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
   * @brief Caluclates regularized inverse standard deviation
   *
   * The output values are calculated as:
   * ```
   * s = sum( (in[pos] - mean[reduced_pos])^2
   * out[reduced_pos] = s > 0 || reg > 0
   *                    ? 1/sqrt(s / reduction_factor + reg^2)
   *                    : 0
   * ```
   * where `reduction_factor` is the number of input elements contributing to a single output.
   *
   * @param ctx     the execution environment
   * @param out     (regularized) inverse standard deviation
   * @param in      input tensor
   * @param mean    mean, used for centering the data
   * @param reg     regularizing term to avoid division by zero (or small numbers);
   *                its squared and added to the sum of squares in variance calculation,
   *                preventing it from being zero; if reg = 0, the results that would
   *                cause division by zero are forced to 0, but small denominators
   *                may still cause problems
   */
  void Run(KernelContext &ctx, const OutListGPU<Out> &out,
           const InListGPU<In> &in, const InListGPU<Mean> &mean, param_t reg = 0);

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
