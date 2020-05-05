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

#include "dali/kernels/reduce/reduce_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_drop_dims.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

/**
 * @brief A position-independent bank, wrapping a functor
 */
template <int non_reduced_ndim, typename Functor>
struct UniformPreprocessorBank {
  DALI_HOST_DEV DALI_FORCEINLINE
  Functor Get(const i64vec<non_reduced_ndim> &) const {
    return {};
  }
};

template <typename Out, typename Scale>
struct ScaleAndConvert {
  Scale scale = 1;
  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    return ConvertSat<Out>(x * scale);
  }
};

template <typename Out, typename In, typename Actual>
class MeanImplBase {
 public:
  Actual &This() { return static_cast<Actual&>(*this); }
  const Actual &This() const { return static_cast<const Actual&>(*this); }

  using scale_t = std::conditional_t<std::is_same<Out, double>::value, double, float>;
  using Postprocessor = ScaleAndConvert<Out, scale_t>;

  Postprocessor *GetPostprocessorsImpl(WorkArea &wa) const {
    assert(!This().ReduceBatch());
    int n = This().SimplifiedOutputShape().num_samples();
    Postprocessor *pp = wa.ParamBuffer<Postprocessor>(n);
    for (int i = 0; i < n; i++) {
      DALI_ENFORCE(This().ReducedElements(i) > 0, "Cannot calculate a mean from 0 elements");
      pp[i].scale = 1.0 / This().ReducedElements(i);
    }
    return pp;
  }

  Postprocessor GetPostprocessorImpl() const {
    assert(This().ReduceBatch());
    DALI_ENFORCE(This().TotalReducedElements() > 0, "Cannot calculate a mean from 0 elements");
    return { static_cast<float>(1.0 / This().TotalReducedElements()) };
  }
};

template <typename Out, typename In, typename Acc = default_sum_acc_t<Out, In>>
class MeanImplGPU : public ReduceImplGPU<Out, In, Acc, MeanImplGPU<Out, In, Acc>>,
                    public MeanImplBase<Out, In, MeanImplGPU<Out, In, Acc>> {
 public:
  using MeanBase = MeanImplBase<Out, In, MeanImplGPU<Out, In, Acc>>;
  using typename MeanBase::Postprocessor;
  reductions::sum GetReduction() const { return {}; }
};

/**
 * @brief Subtracts a mean value stored in specified memory location
 */
template <class Mean>
struct SubtractMeanIndirect {
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

  using Preprocessor = SubtractMeanIndirect<Mean>;

  static_assert(sizeof(Preprocessor) == sizeof(Mean*),
    "A variance functor must carry only a pointer to the mean");

  template <int non_reduced_dims>
  using PreprocessorBank = VariancePreprocessor<non_reduced_dims, Mean>;

  void InitMean(const InListGPU<Mean> &mean) {
    mean_ = reshape(mean, This().SimplifiedOutputShape());
  }

  Preprocessor GetPreprocessorImpl() const {
    assert(This().SimplifiedOutputShape().num_elements() == 1);
    return Preprocessor { mean_.data[0] };
  }

  Preprocessor *GetPreprocessorsImpl(WorkArea &wa) const {
    int n = This().SimplifiedInputShape().num_samples();
    assert(This().SimplifiedOutputShape().num_elements() ==
           This().SimplifiedOutputShape().num_samples());
    Preprocessor *pp = wa.ParamBuffer<Preprocessor>(n);
    for (int i = 0; i < n; i++) {
      int o = This().ReduceBatch() ? 0 : i;
      pp[i] = { mean_.data[o] };
    }
    return pp;
  }

  PreprocessorBank<1> *
  GetPreprocessorBanks(WorkArea &wa, int axis, std::integral_constant<int, 1>) const {
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
  GetPreprocessorBanks(WorkArea &wa, int axis, std::integral_constant<int, 2>) const {
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
      bank.inner_dims = DropDims(inner_shape, mask);  // reindexing, if necessary
    }
    return banks;
  }

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *
  GetPreprocessorBanksImpl(WorkArea &wa, int axis) const {
    return GetPreprocessorBanks(wa, axis, std::integral_constant<int, non_reduced_dims>());
  }
};

template <typename Out, typename Scale>
struct ScaleSqrtConvert {
  Scale scale = 1;

  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    return ConvertSat<Out>(sqrt(x * scale));
  }
};

template <typename Out, typename In, typename Actual>
class RootMeanImplBase {
 public:
  Actual &This() { return static_cast<Actual&>(*this); }
  const Actual &This() const { return static_cast<const Actual&>(*this); }

  using scale_t = std::conditional_t<std::is_same<Out, double>::value, double, float>;
  using Postprocessor = ScaleSqrtConvert<Out, scale_t>;

  Postprocessor *GetPostprocessorsImpl(WorkArea &wa) const {
    assert(!This().ReduceBatch());
    int n = This().SimplifiedOutputShape().num_samples();
    Postprocessor *pp = wa.ParamBuffer<Postprocessor>(n);
    for (int i = 0; i < n; i++) {
      DALI_ENFORCE(This().ReducedElements(i) > 0, "Cannot calculate a mean from 0 elements");
      pp[i].scale = 1.0 / This().ReducedElements(i);
    }
    return pp;
  }

  Postprocessor GetPostprocessorImpl() const {
    assert(This().ReduceBatch());
    DALI_ENFORCE(This().TotalReducedElements() > 0, "Cannot calculate a mean from 0 elements");
    return { static_cast<float>(1.0 / This().TotalReducedElements()) };
  }
};


template <typename Out, typename In, typename Acc =
  default_sum_acc_t<Out, decltype(reductions::square()(In()))>>
class RootMeanSquareImplGPU
    : public ReduceImplGPU<Out, In, Acc, RootMeanSquareImplGPU<Out, In, Acc>>
    , public RootMeanImplBase<Out, In, RootMeanSquareImplGPU<Out, In, Acc>> {
 public:
  using MeanBase = RootMeanImplBase<Out, In, RootMeanSquareImplGPU<Out, In, Acc>>;
  using typename MeanBase::Postprocessor;

  using Preprocessor = reductions::square;
  template <int non_reduced_dims>
  using PreprocessorBank = UniformPreprocessorBank<non_reduced_dims, Preprocessor>;

  Preprocessor GetPreprocessorImpl() const { return {}; }
  Preprocessor *GetPreprocessorsImpl(WorkArea &wa) const { return nullptr; }

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *GetPreprocessorBanksImpl(WorkArea &wa, int axis) const {
    return nullptr;
  }

  reductions::sum GetReduction() const { return {}; }
};

template <typename Out, typename In, typename Mean = Out, typename Acc = Out>
class StdDevImplGPU : public ReduceImplGPU<Out, In, Acc, StdDevImplGPU<Out, In, Mean, Acc>>,
                      public VarianceImplBase<Out, In, Mean, StdDevImplGPU<Out, In, Mean, Acc>>,
                      public RootMeanImplBase<Out, In, StdDevImplGPU<Out, In, Mean, Acc>> {
 public:
  using ReduceBase = ReduceImplGPU<Out, In, Acc, StdDevImplGPU<Out, In, Mean, Acc>>;
  using VarBase = VarianceImplBase<Out, In, Mean, StdDevImplGPU<Out, In, Mean, Acc>>;
  using MeanBase = RootMeanImplBase<Out, In, StdDevImplGPU<Out, In, Mean, Acc>>;

  using Preprocessor = typename VarBase::Preprocessor;
  template <int non_reduced_dims>
  using PreprocessorBank = typename VarBase::template PreprocessorBank<non_reduced_dims>;

  using Postprocessor = typename MeanBase::Postprocessor;
  reductions::sum GetReduction() const { return {}; }

  void Run(KernelContext &kctx,
           const OutListGPU<Out> &out,
           const InListGPU<In> &in,
           const InListGPU<Mean> &mean) {
    this->InitMean(mean);
    ReduceBase::Run(kctx, out, in);
  }
};

template <typename Out, typename ScaleAndReg>
struct RegularizedInvSqrt {
  ScaleAndReg scale = 1, reg = 0;

  template <typename T>
  DALI_HOST_DEV Out operator()(T x) const {
    float s = scale * (x + reg);
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

  void SetRegularizationTerm(param_t reg) {
    regularization_ = reg * reg;
  }

  param_t regularization_ = 0.0f;

  Postprocessor *GetPostprocessorsImpl(WorkArea &wa) const {
    assert(!This().ReduceBatch());
    int n = This().SimplifiedOutputShape().num_samples();
    Postprocessor *pp = wa.ParamBuffer<Postprocessor>(n);
    for (int i = 0; i < n; i++) {
      DALI_ENFORCE(This().ReducedElements(i) > 0, "Cannot calculate a mean from 0 elements");
      pp[i].scale = param_t(1.0 / This().ReducedElements(i));
      pp[i].reg = regularization_;
    }
    return pp;
  }

  Postprocessor GetPostprocessorImpl() const {
    assert(This().ReduceBatch());
    DALI_ENFORCE(This().TotalReducedElements() > 0, "Cannot calculate a mean from 0 elements");
    return { param_t(1.0f / This().TotalReducedElements()), regularization_ };
  }
};

template <typename Out, typename In, typename Mean = Out, typename Acc = Out>
class InvStdDevImplGPU :
      public ReduceImplGPU<Out, In, Acc, InvStdDevImplGPU<Out, In, Mean, Acc>>,
      public VarianceImplBase<Out, In, Mean, InvStdDevImplGPU<Out, In, Mean, Acc>>,
      public RegularizedInvRMS<Out, In, InvStdDevImplGPU<Out, In, Mean, Acc>> {
 public:
  using ReduceBase = ReduceImplGPU<Out, In, Acc, InvStdDevImplGPU<Out, In, Mean, Acc>>;
  using VarBase = VarianceImplBase<Out, In, Mean, InvStdDevImplGPU<Out, In, Mean, Acc>>;
  using MeanBase = RegularizedInvRMS<Out, In, InvStdDevImplGPU<Out, In, Mean, Acc>>;

  using Preprocessor = typename VarBase::Preprocessor;
  template <int non_reduced_dims>
  using PreprocessorBank = typename VarBase::template PreprocessorBank<non_reduced_dims>;

  using Postprocessor = typename MeanBase::Postprocessor;
  reductions::sum GetReduction() const { return {}; }

  void Run(KernelContext &kctx,
           const OutListGPU<Out> &out,
           const InListGPU<In> &in,
           const InListGPU<Mean> &mean,
           float reg = 0.0f) {
    this->InitMean(mean);
    this->SetRegularizationTerm(reg);
    ReduceBase::Run(kctx, out, in);
  }
};

}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_MEAN_STDDEV_GPU_IMPL_CUH_
