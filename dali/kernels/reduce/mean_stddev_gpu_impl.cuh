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

#ifndef DALI_KERNELS_REDUCE_MEAN_STDDEV_GPU_IMPL_CUH_
#define DALI_KERNELS_REDUCE_MEAN_STDDEV_GPU_IMPL_CUH_

/**
 * @file
 *
 * This file contains the classes needed to implement reductions with pre-
 * and postprocessing: mean, root mean square, standard deviation (and its reciprocal).
 */

#include "dali/kernels/reduce/reduce_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_drop_dims.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

template <typename T>
using scale_t = std::conditional_t<std::is_same<T, double>::value, double, float>;

template <typename Out, typename Scale = scale_t<Out>>
struct ScaleAndConvert {
  using scale_t = Scale;
  scale_t scale = 1;

  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    return ConvertSat<Out>(x * scale);
  }
};


template <typename Out, typename Scale = scale_t<Out>>
struct ScaleSqrtConvert {
  using scale_t = Scale;
  scale_t scale = 1;

  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    return ConvertSat<Out>(sqrt(x * scale));
  }
};

template <typename Out, typename In, typename Actual,
          typename Postprocessor = ScaleAndConvert<Out>>
class MeanImplBase {
 public:
  Actual &This() { return static_cast<Actual&>(*this); }
  const Actual &This() const { return static_cast<const Actual&>(*this); }

  using postprocessor_t = Postprocessor;
  using scale_t = typename Postprocessor::scale_t;

  Postprocessor GetPostprocessorImpl(int sample_index, bool reduce_batch) const {
    int64_t reduced_elems = reduce_batch ? This().TotalReducedElements()
                                         : This().ReducedElements(sample_index);
    return GetPostprocessorImpl(reduced_elems, 0);
  }

  Postprocessor GetPostprocessorImpl(int64_t reduced_elems, int ddof) const {
    DALI_ENFORCE(reduced_elems > 0, "Cannot calculate a mean from 0 elements");
    auto denominator = reduced_elems - ddof;
    return { denominator > 0 ? scale_t(1.0 / denominator) : 0 };
  }
};

template <typename Out, typename In, typename Actual>
using RootMeanImplBase = MeanImplBase<Out, In, Actual, ScaleSqrtConvert<Out>>;

/**
 * @brief Implements mean reduction
 */
template <typename Out, typename In, typename Acc = default_sum_acc_t<Out, In>>
class MeanImplGPU : public ReduceImplGPU<Out, In, Acc, MeanImplGPU<Out, In, Acc>>,
                    public MeanImplBase<Out, In, MeanImplGPU<Out, In, Acc>> {
 public:
  reductions::sum GetReduction() const { return {}; }
};


/**
 * @brief Implements mean square reduction
 */
template <typename Out, typename In, typename Acc =
  default_sum_acc_t<Out, decltype(reductions::square()(In()))>>
class MeanSquareImplGPU
    : public ReduceImplGPU<Out, In, Acc, MeanSquareImplGPU<Out, In, Acc>>
    , public MeanImplBase<Out, In, MeanSquareImplGPU<Out, In, Acc>> {
 public:
  using Preprocessor = reductions::square;
  template <int non_reduced_dims>
  using PreprocessorBank = UniformPreprocessorBank<non_reduced_dims, Preprocessor>;

  Preprocessor GetPreprocessorImpl(int sample_idx, bool batch) const { return {}; }

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *
  GetPreprocessorBanksImpl(WorkArea &wa, int axis, int_const<non_reduced_dims>) const {
    return nullptr;
  }

  reductions::sum GetReduction() const { return {}; }
};

/**
 * @brief Implements root mean square reduction
 */
template <typename Out, typename In, typename Acc =
  default_sum_acc_t<Out, decltype(reductions::square()(In()))>>
class RootMeanSquareImplGPU
    : public ReduceImplGPU<Out, In, Acc, RootMeanSquareImplGPU<Out, In, Acc>>
    , public RootMeanImplBase<Out, In, RootMeanSquareImplGPU<Out, In, Acc>> {
 public:
  using Preprocessor = reductions::square;
  template <int non_reduced_dims>
  using PreprocessorBank = UniformPreprocessorBank<non_reduced_dims, Preprocessor>;

  Preprocessor GetPreprocessorImpl(int sample_idx, bool batch) const { return {}; }

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *
  GetPreprocessorBanksImpl(WorkArea &wa, int axis, int_const<non_reduced_dims>) const {
    return nullptr;
  }

  reductions::sum GetReduction() const { return {}; }
};


/**
 * @brief Subtracts a mean value stored in specified memory location and squares the difference
 *
 * This postprocessor is necessary because regular `variance` would require gathering means
 * for all samples, which may be scattered in non-contiguous device memory.
 */
template <class Mean>
struct VarianceIndirect {
  const Mean *__restrict__ mean = nullptr;
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE
  auto operator()(const T &x) const noexcept {
    #ifdef __CUDA_ARCH__
      auto d = x - __ldg(mean);
    #else
      auto d = x - *mean;
    #endif
      return d * d;
  }
};

/**
 * @brief A preprocessor bank which returns a `reduce::variance` functor with
 *        mean value taken from a tensor
 */
template <int non_reduced_ndim, typename Mean>
struct VariancePreprocessorBank;

template <typename Mean>
struct VariancePreprocessorBank<1, Mean> {
  const Mean *__restrict__ mean;
  i64vec<1> stride;

  DALI_HOST_DEV DALI_FORCEINLINE
  reductions::variance<Mean> Get(const i64vec<1> &pos) const {
    auto offset = dot(pos, stride);
  #ifdef __CUDA_ARCH__
    Mean m = __ldg(mean + offset);
  #else
    Mean m = mean[offset];
  #endif
    return { m };
  }
};

template <typename Mean>
struct VariancePreprocessorBank<2, Mean> {
  const Mean *mean;
  i64vec<2> stride;
  /// Calculates the fully reduced inner offset based on non-reduced `pos[1]`
  DropDims<3> inner_dims;

  DALI_HOST_DEV DALI_FORCEINLINE
  reductions::variance<Mean> Get(const i64vec<2> &pos) const {
    auto offset = dot(i64vec2(pos[0], inner_dims.reindex(pos[1])), stride);
  #ifdef __CUDA_ARCH__
    Mean m = __ldg(mean + offset);
  #else
    Mean m = mean[offset];
  #endif
    return { m };
  }
};

template <typename Out, typename In, typename Mean, typename Actual>
class VarianceImplBase {
 public:
  Actual &This() { return static_cast<Actual&>(*this); }
  const Actual &This() const { return static_cast<const Actual&>(*this); }

  void SetMean(const InListGPU<Mean> &mean, cudaStream_t stream) {
    mean_ = mean;
    mean_.reshape(This().SimplifiedOutputShape());
  }

  InListGPU<Mean> mean_;

  using Preprocessor = VarianceIndirect<Mean>;

  static_assert(sizeof(Preprocessor) == sizeof(Mean*),
    "A variance functor must carry only a pointer to the mean");

  template <int non_reduced_dims>
  using PreprocessorBank = VariancePreprocessorBank<non_reduced_dims, Mean>;

  void InitMean(const InListGPU<Mean> &mean) {
    mean_ = reshape(mean, This().SimplifiedOutputShape(), true);
  }

  Preprocessor GetPreprocessorImpl(int sample_index, bool batch) const {
    assert(sample_index < This().SimplifiedOutputShape().num_samples());
    return Preprocessor { mean_.data[sample_index] };
  }

  PreprocessorBank<1> *
  GetPreprocessorBanks(WorkArea &wa, int axis, int_const<1>) const {
    using Bank = PreprocessorBank<1>;
    int n = This().SimplifiedInputShape().num_samples();
    Bank *banks = wa.ParamBuffer<Bank>(n);

    for (int i = 0; i < n; i++) {
      int o = This().ReduceBatch() ? 0 : i;
      auto shape = This().SimplifiedOutputShape().tensor_shape_span(o);
      auto &bank = banks[i];
      bank.mean = mean_.data[o];
      bank.stride[0] = volume(shape.begin() + axis, shape.end());  // outer stride
    }
    return banks;
  }

  PreprocessorBank<2> *
  GetPreprocessorBanks(WorkArea &wa, int axis, int_const<2>) const {
    using Bank = PreprocessorBank<2>;
    int n = This().SimplifiedInputShape().num_samples();
    Bank *banks = wa.ParamBuffer<Bank>(n);

    SmallVector<int, 6> remaining_axes;
    for (int a : This().SimplifiedAxes())
      if (a > axis)
        remaining_axes.push_back(a - axis - 1);
    int mask = to_bit_mask(remaining_axes);

    for (int i = 0; i < n; i++) {
      int o = This().ReduceBatch() ? 0 : i;
      auto &bank = banks[i];
      auto in_shape = This().SimplifiedInputShape().tensor_shape_span(i);
      auto out_shape = This().SimplifiedOutputShape().tensor_shape_span(o);
      auto inner_shape = span<const int64_t>(in_shape.begin() + axis + 1, in_shape.end());
      bank.mean = mean_.data[o];
      bank.stride[0] = volume(out_shape.begin() + axis, out_shape.end());  // outer stride
      bank.stride[1] = 1;  // inner stride, always 1?
      bank.inner_dims = DropDims<3>(inner_shape, mask);  // reindexing, if necessary
    }
    return banks;
  }

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *
  GetPreprocessorBanksImpl(WorkArea &wa, int axis, int_const<non_reduced_dims> nrd) const {
    return GetPreprocessorBanks(wa, axis, nrd);
  }
};

/**
 * @brief Implements variance with externally provided mean
 */
template <typename Out, typename In, typename Mean = Out, typename Acc = Out>
class VarianceImplGPU : public ReduceImplGPU<Out, In, Acc, VarianceImplGPU<Out, In, Mean, Acc>>,
                      public VarianceImplBase<Out, In, Mean, VarianceImplGPU<Out, In, Mean, Acc>>,
                      public MeanImplBase<Out, In, VarianceImplGPU<Out, In, Mean, Acc>> {
 public:
  using ReduceBase = ReduceImplGPU<Out, In, Acc, VarianceImplGPU<Out, In, Mean, Acc>>;
  using MeanBase = MeanImplBase<Out, In, VarianceImplGPU<Out, In, Mean, Acc>>;

  reductions::sum GetReduction() const { return {}; }

  typename MeanBase::postprocessor_t
  GetPostprocessorImpl(int sample_index, bool reduce_batch) const {
    int64_t reduced_elems = reduce_batch ? this->TotalReducedElements()
                                         : this->ReducedElements(sample_index);
    return MeanBase::GetPostprocessorImpl(reduced_elems, ddof_);
  }

  void Run(KernelContext &kctx,
           const OutListGPU<Out> &out,
           const InListGPU<In> &in,
           const InListGPU<Mean> &mean,
           int ddof = 0) {
    ddof_ = ddof;
    this->InitMean(mean);
    ReduceBase::Run(kctx, out, in);
  }

 private:
  int ddof_ = 0;
};

/**
 * @brief Implements standard deviation with externally provided mean
 */
template <typename Out, typename In, typename Mean = Out, typename Acc = Out>
class StdDevImplGPU : public ReduceImplGPU<Out, In, Acc, StdDevImplGPU<Out, In, Mean, Acc>>,
                      public VarianceImplBase<Out, In, Mean, StdDevImplGPU<Out, In, Mean, Acc>>,
                      public RootMeanImplBase<Out, In, StdDevImplGPU<Out, In, Mean, Acc>> {
 public:
  using ReduceBase = ReduceImplGPU<Out, In, Acc, StdDevImplGPU<Out, In, Mean, Acc>>;
  using RMSBase = RootMeanImplBase<Out, In, StdDevImplGPU<Out, In, Mean, Acc>>;

  reductions::sum GetReduction() const { return {}; }

  typename RMSBase::postprocessor_t
  GetPostprocessorImpl(int sample_index, bool reduce_batch) const {
    int64_t reduced_elems = reduce_batch ? this->TotalReducedElements()
                                         : this->ReducedElements(sample_index);
    return RMSBase::GetPostprocessorImpl(reduced_elems, ddof_);
  }

  void Run(KernelContext &kctx,
           const OutListGPU<Out> &out,
           const InListGPU<In> &in,
           const InListGPU<Mean> &mean,
           int ddof = 0) {
    ddof_ = ddof;
    this->InitMean(mean);
    ReduceBase::Run(kctx, out, in);
  }

 private:
  int ddof_ = 0;
};

template <typename Out, typename ScaleAndReg>
struct RegularizedInvSqrt {
  ScaleAndReg scale = 1, reg = 0;

  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    float s = scale * x + reg;
    return s ? ConvertSat<Out>(rsqrt(s)) : Out(0);
  }
};

template <typename Out, typename In, typename Actual>
class RegularizedInvRMS {
 public:
  Actual &This() { return static_cast<Actual&>(*this); }
  const Actual &This() const { return static_cast<const Actual&>(*this); }

  using param_t = std::conditional_t<std::is_same<Out, double>::value, double, float>;
  using Postprocessor = RegularizedInvSqrt<Out, param_t>;

  void SetStdDevParams(int ddof, param_t epsilon) {
    if (!(epsilon >= 0))  // >= 0 and not NaN
      throw std::range_error("The regularizing term must be a non-negative number.");
    if (ddof < 0)
      throw std::range_error("Delta Degrees of Freedom must be a non-negative number.");
    regularization_ = epsilon;
    ddof_ = ddof;
  }

  param_t regularization_ = 0.0f;
  int     ddof_ = 0;

  Postprocessor GetPostprocessorImpl(int sample_index, bool reduce_batch) const {
    int64_t reduced_elems = reduce_batch ? This().TotalReducedElements()
                                         : This().ReducedElements(sample_index);
    DALI_ENFORCE(reduced_elems > 0, "Cannot calculate a mean from 0 elements");
    param_t scale = reduced_elems > ddof_ ? param_t(1.0 / (reduced_elems - ddof_)) : 0;
    return { scale, regularization_ };
  }
};

/**
 * @brief Implements regularized inverse standard  deviation reduction with externally provided mean
 */
template <typename Out, typename In, typename Mean = Out, typename Acc = Out>
class InvStdDevImplGPU :
      public ReduceImplGPU<Out, In, Acc, InvStdDevImplGPU<Out, In, Mean, Acc>>,
      public VarianceImplBase<Out, In, Mean, InvStdDevImplGPU<Out, In, Mean, Acc>>,
      public RegularizedInvRMS<Out, In, InvStdDevImplGPU<Out, In, Mean, Acc>> {
 public:
  using ReduceBase = ReduceImplGPU<Out, In, Acc, InvStdDevImplGPU<Out, In, Mean, Acc>>;

  reductions::sum GetReduction() const { return {}; }

  /**
   *
   */
  void Run(KernelContext &kctx,
           const OutListGPU<Out> &out,
           const InListGPU<In> &in,
           const InListGPU<Mean> &mean,
           int ddof = 0,
           float epsilon = 0.0f) {
    this->InitMean(mean);
    this->SetStdDevParams(ddof, epsilon);
    ReduceBase::Run(kctx, out, in);
  }
};

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_MEAN_STDDEV_GPU_IMPL_CUH_
