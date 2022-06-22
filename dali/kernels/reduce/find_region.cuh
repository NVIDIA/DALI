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

#ifndef DALI_KERNELS_REDUCE_FIND_REGION_CUH_
#define DALI_KERNELS_REDUCE_FIND_REGION_CUH_

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/reduce/reduce_gpu_impl.cuh"
#include "dali/kernels/reduce/reduce_drop_dims.h"

namespace dali {
namespace kernels {

/**
 * @brief Preprocess input for a minimum reduction, each element contains two values
 *        {i, -i}               if predicate(sample[i])
 *        max_value, max_value  otherwise.
 *        Later, this is reduced so that the first value represents the index of the
 *        first element that satisfies the predicate, and the second value represents
 *        the negative of the index of the last value that satisfies the predicate.
 */
template <typename T, typename Predicate>
struct FindRegionPreprocess {
  const Predicate *__restrict__ predicate = nullptr;
  const T *__restrict__ base = nullptr;

  DALI_HOST_DEV DALI_FORCEINLINE i64vec2 operator()(const T &x) const noexcept {
    if (!predicate || (*predicate)(x)) {
      return {&x - base, base - &x};
    } else {
      return i64vec2(max_value<int64_t>());
    }
  }
};

/**
 * @brief Converts the reduced {min_i, -max_i} pair to a {begin, end} representation
 */
struct FindRegionPostprocess {
  DALI_HOST_DEV DALI_FORCEINLINE i64vec2 operator()(i64vec2 acc) const {
    if (acc == i64vec2(max_value<int64_t>()))
      return i64vec2(0);  // empty region
    else
      // begin = min_i, end = -(-max_i) + 1 = max_i + 1
      return i64vec2(acc.x, -acc.y + 1);
  }
};

/**
 * @brief Finds region (begin, end) satisfying a predicate
 */
template <typename In, typename Predicate>
class FindRegionGPU
    : public reduce_impl::ReduceImplGPU<i64vec2, In, i64vec2,
                                        FindRegionGPU<In, Predicate>> {
 public:
  using Reduction = reductions::min;
  using ReduceBase =
      reduce_impl::ReduceImplGPU<i64vec2, In, i64vec2, FindRegionGPU<In, Predicate>>;
  using Preprocessor = FindRegionPreprocess<In, Predicate>;
  using Postprocessor = FindRegionPostprocess;

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

  template <int non_reduced_dims>
  using PreprocessorBank = reduce_impl::UniformPreprocessorBank<non_reduced_dims, Preprocessor>;

  template <int non_reduced_dims>
  PreprocessorBank<non_reduced_dims> *GetPreprocessorBanksImpl(
      reduce_impl::WorkArea &wa, int axis, reduce_impl::int_const<non_reduced_dims>) const {
    return nullptr;
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

  void Run(KernelContext &kctx, const OutListGPU<i64vec2, 0> &out, const InListGPU<In, 1> &in,
           const InListGPU<Predicate, 0> &predicates) {
    this->InitSampleData(in);
    this->InitPredicates(predicates);
    ReduceBase::Run(kctx, out, in);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_FIND_REGION_CUH_
