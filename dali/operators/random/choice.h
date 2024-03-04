// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_RANDOM_CHOICE_H_
#define DALI_OPERATORS_RANDOM_CHOICE_H_

#include <random>
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/operators/random/rng_base_gpu.h"
#include "dali/operators/random/rng_base_cpu.h"

#define DALI_COINFLIP_TYPES bool, uint8_t, int32_t

namespace dali {

struct ChoiceSampleDist {
  using DistType = std::discrete_distribution<>;

  DALI_HOST_DEV explicit ChoiceSampleDist() {}

  // TODO(klecki): We need to initialize once or many tiles based on returns_, just
  // handle the copy in the constructor once, accept begin and end here.
  DALI_HOST_DEV explicit ChoiceSampleDist(std::vector<float> p_dist, bool returns)
    : p_dist_(std::move(p_dist)), dist_(p_dist_.begin(), p_dist_.end()), returns_(returns) {}

  template <typename Generator>
  DALI_HOST_DEV int Generate(Generator &st) {
    if (!returns_) {
      auto val = dist_(st);
      p_dist_[val] = 0;
      dist_ = DistType(p_dist_.begin(), p_dist_.end());
      return val;
    }
    return dist_(st);
  }
  DistType dist_;
  // TODO(klecki): use it only when returns_ == false
  std::vector<float> p_dist_;
  bool returns_;
};

template <typename Backend>
class Choice : public rng::RNGBase<Backend, Choice<Backend>, false> {
 public:
  using BaseImpl = rng::RNGBase<Backend, Choice<Backend>, false>;

  using Impl = ChoiceSampleDist;

  explicit Choice(const OpSpec &spec)
      : BaseImpl(spec),
        p_dist_("p", spec) {
  }

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    p_dist_.Acquire(spec, ws, nsamples);
  }

  DALIDataType DefaultDataType() const {
    return DALI_INT32;
  }

  template <typename T>
  bool SetupDists(Impl* dists_data, int nsamples) {
    for (int s = 0; s < nsamples; s++) {
      std::vector<float> p_dist(p_dist_[s].data, p_dist_[s].data + p_dist_[s].num_elements());
      dists_data[s] = Impl{p_dist_[s].data, p_dist_[s].data + p_dist_[s].num_elements()};
    }
    return true;
  }

  using BaseImpl::RunImpl;
  void RunImpl(Workspace &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_COINFLIP_TYPES), (
      BaseImpl::template RunImplTyped<T, Impl>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is currently not supported. "
                            "Supported types are : ", ListTypeNames<DALI_COINFLIP_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Operator<Backend>::max_batch_size_;
  using BaseImpl::dtype_;
  using BaseImpl::backend_data_;

  ArgValue<float, 1> p_dist_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_CHOICE_H_
