// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_REDUCE_FIND_REDUCE_CUH_
#define DALI_KERNELS_REDUCE_FIND_REDUCE_CUH_

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/geom/vec.h"

#include "dali/kernels/reduce/reduce_drop_dims.h"
#include "dali/kernels/reduce/reduce_gpu_impl.cuh"

namespace dali {
namespace kernels {

/**
 * @brief Preprocess step that returns the index of the sample if the value satisfies the predicate
 *        The neutral value (the "not found" value) will depend on the reduction that we need to apply.
 */
template <typename T, typename Predicate, typename Reduction, typename Acc = ptrdiff_t>
struct FindReducePreprocess {
  const Predicate *__restrict__ predicate = nullptr;
  const T *__restrict__ base = nullptr;

  DALI_HOST_DEV DALI_FORCEINLINE Acc operator()(const T &x) const noexcept {
    if (!predicate || (*predicate)(x)) {
      return &x - base;
    } else {
      return Reduction::template neutral<Acc>();
    }
  }
};

/**
 * @brief Postprocess step required to convert the min/max neutral value to -1
          (meaning "not found")
 */
template <typename Out, typename Reduction>
struct FindReducePostprocess {
  template <typename T>
  DALI_HOST_DEV DALI_FORCEINLINE Out operator()(T x) const {
    return x == Reduction::template neutral<T>() ? Out(-1) : x;
  }
};

/**
 * @brief Finds positions in the input that satisfy a predicate, reducing them
 *        to a single index (e.g. first, last)
 */
template <typename Out, typename In, typename Predicate, typename Reduction,
          typename Acc = ptrdiff_t>
class FindReduceGPU
    : public reduce_impl::ReduceImplGPU<Out, In, Acc,
                                        FindReduceGPU<Out, In, Predicate, Reduction, Acc>> {
 public:
  using ReduceBase =
      reduce_impl::ReduceImplGPU<Out, In, Acc, FindReduceGPU<Out, In, Predicate, Reduction, Acc>>;
  using Preprocessor = FindReducePreprocess<In, Predicate, Reduction, Acc>;
  using Postprocessor = FindReducePostprocess<Out, Reduction>;

  std::vector<const In *> base_;
  InListGPU<Predicate, 0> predicates_;

  void InitSampleData(const InListGPU<In> &in) {
    int nsamples = in.num_samples();
    base_.resize(nsamples);
    for (int i = 0; i < nsamples; i++) {
      base_[i] = in[i].data;
    }
  }

  void InitPredicates(const InListGPU<Predicate, 0> &predicates) {
    predicates_ = predicates;
  }

  Preprocessor GetPreprocessorImpl(int sample_idx, bool batch) const {
    assert(!batch);  // not allowed
    return Preprocessor{predicates_[sample_idx].data, base_[sample_idx]};
  }

  Postprocessor GetPostprocessorImpl(int sample_index, bool batch) const {
    assert(!batch);  // not allowed
    return {};
  }

  Reduction GetReduction() const {
    return {};
  }

  void Setup(KernelContext &kctx, const TensorListShape<1> &in_shape) {
    std::array<int, 1> axes = {0};
    ReduceBase::Setup(kctx, in_shape, make_cspan(axes), false, false);
  }

  void Run(KernelContext &kctx, const OutListGPU<Out, 0> &out, const InListGPU<In, 1> &in,
           const InListGPU<Predicate, 0> &predicates) {
    this->InitSampleData(in);
    this->InitPredicates(predicates);
    ReduceBase::Run(kctx, out, in);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_FIND_REDUCE_CUH_
