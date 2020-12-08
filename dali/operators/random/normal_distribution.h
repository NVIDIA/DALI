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

#ifndef DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_
#define DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_

#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_NORMDIST_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, \
                             int64_t, float16, float, double)

namespace dali {

template <typename Backend, typename Impl>
class NormalDistribution : public RNGBase<Backend, Impl> {
 public:
  explicit NormalDistribution(const OpSpec &spec)
      : RNGBase<Backend, Impl>(spec),
        mean_("mean", spec),
        stddev_("stddev", spec) {}

  ~NormalDistribution() override = default;

  void AcquireArgs(const OpSpec &spec, const workspace_t<Backend> &ws, int nsamples) {
    mean_.Acquire(spec, ws, nsamples, true);
    stddev_.Acquire(spec, ws, nsamples, true);
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_NORMDIST_TYPES, (
      this->template RunImplTyped<T>(ws);
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

 protected:
  using RNGBase<Backend, Impl>::dtype_;

  ArgValue<float> mean_;
  ArgValue<float> stddev_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_NORMAL_DISTRIBUTION_H_
