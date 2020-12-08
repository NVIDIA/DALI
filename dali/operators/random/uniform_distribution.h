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

#ifndef DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_
#define DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_UNIFORM_DIST_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, \
                                 int64_t, float16, float, double)

namespace dali {

template <typename Backend, typename Impl>
class UniformDistribution : public RNGBase<Backend, Impl> {
 public:
  explicit UniformDistribution(const OpSpec &spec)
      : RNGBase<Backend, Impl>(spec),
        values_("values", spec),
        range_("range", spec) {}

  ~UniformDistribution() override = default;

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    if (values_.IsDefined()) {
      values_.Acquire(spec, ws, nsamples, false);
    } else {
      range_.Acquire(spec, ws, nsamples, TensorShape<1>{2});
    }
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_UNIFORM_DIST_TYPES, (
      if (values_.IsDefined()) {
        using Dist = typename Impl::DistDiscrete;
        this->template RunImplTyped<T, Dist>(ws);
      } else {
        using Dist = typename Impl::template Dist<T>::type;
        this->template RunImplTyped<T, Dist>(ws);
      }
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 protected:
  using RNGBase<Backend, Impl>::dtype_;

  ArgValue<float, 1> values_;
  ArgValue<float, 1> range_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_UNIFORM_DISTRIBUTION_H_
