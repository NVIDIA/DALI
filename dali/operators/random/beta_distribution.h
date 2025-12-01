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

#ifndef DALI_OPERATORS_RANDOM_BETA_DISTRIBUTION_H_
#define DALI_OPERATORS_RANDOM_BETA_DISTRIBUTION_H_

#include <random>
#include <vector>
#include "dali/operators/random/rng_base.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_BETA_DIST_TYPES float, double

namespace dali {

template <typename T>
struct BetaDistributionImpl {
  BetaDistributionImpl() : BetaDistributionImpl{1, 1} {}

  explicit BetaDistributionImpl(T alpha, T beta)
      : has_small_param_{alpha < 1 && beta < 1},
        alpha_{alpha},
        beta_{beta},
        b_div_a_{beta_ / alpha_},
        a_{has_small_param_ ? alpha_ + 1 : alpha_},
        b_{has_small_param_ ? beta_ + 1 : beta_},
        exp_{} {}

  template <typename Generator>
  T Generate(Generator &st) {
    return has_small_param_ ? GenerateGammaExp(st) : GenerateGamma(st);
  }

 private:
  template <typename Generator>
  T GenerateGamma(Generator &st) {
    // https://en.wikipedia.org/wiki/Beta_distribution#Random_variate_generation
    T a = a_(st);
    T b = b_(st);
    return a / (a + b);
  }

  template <typename Generator>
  T GenerateGammaExp(Generator &st) {
    assert(alpha_ < 1 && beta_ < 1);
    // For alpha >= 1 and x in [0, 1], Gamma(alpha).cdf(x) <= Exp(1).cdf(x),
    // i.e., the probability of sampling [0, x] is less than 1 - exp(-x) <= x,
    // so the probability of the denominator in `A / (A + B)` being rounded to zero
    // is negligable.
    // However, for alpha < 1, the gamma distribution has high density near 0.
    // Here, we use the fact that for X ~ Gamma(alpha + 1), U ~ Uniform(0, 1)^(1/alpha),
    // the `A = X * U` has Gamma(alpha) distribution (see Luc Devroye, Non-Uniform Random
    // Variate Generation, p. 182)
    // We can compute A / (A + B) = X * U_a / (X * U_a + Y * U_b) =
    // (X * U_a / max(U_a, U_b)) / (X * U_a / max(U_a, U_b) + Y * U_b / max(U_a, U_b)).
    // This way, we have either X or Y in the denominator sampled from "safe" Gamma
    // with params > 1.
    // The Uniform(0, 1)^(1/alpha) is concentrated near zero, for this reason
    // the U_a / max(U_a, U_b), U_b / max(U_a, U_b) are computed in logarithmic scale.
    T a = a_(st);
    T b = b_(st);
    // By inverse transform sampling, the ln(Uniform(0, 1)) = -Exp(1).
    T ln_ua = -exp_(st);
    T ln_ub = -exp_(st);
    // -Exp_a / alpha_ < -Exp_b / beta_, iff -Exp_a * (beta_/alpha_) < -Exp_b
    // but in that form, we can handle subnormal alpha or beta, that would
    // result in two inifnities otherwise
    if (ln_ua * b_div_a_ < ln_ub) {
      // ln_ua / alpha - ln_ub / beta
      T c = (ln_ua * b_div_a_ - ln_ub) / beta_;  // [-inf, 0]
      T ac = a * std::exp(c);                    // a * [0, 1]
      return ac / (ac + b);
    } else {
      // ln_ub / beta - ln_ua / alpha
      T c = (ln_ub - ln_ua * b_div_a_) / beta_;  // [-inf, 0]
      T bc = b * std::exp(c);                    // b * [0, 1]
      return a / (a + bc);
    }
  }

  bool has_small_param_;
  T alpha_, beta_, b_div_a_;
  std::gamma_distribution<T> a_;
  std::gamma_distribution<T> b_;
  std::exponential_distribution<T> exp_;
};

template <typename Backend>
class BetaDistribution : public rng::RNGBase<Backend, BetaDistribution<Backend>, false> {
 public:
  using Base = rng::RNGBase<Backend, BetaDistribution<Backend>, false>;
  static_assert(std::is_same_v<Backend, CPUBackend>, "GPU backend is not implemented");

  explicit BetaDistribution(const OpSpec &spec)
      : Base(spec), alpha_("alpha", spec), beta_("beta", spec) {}

  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {
    // read only once for build time arguments
    if (alpha_.HasArgumentInput() || !alpha_.size()) {
      alpha_.Acquire(spec, ws, alpha_.HasArgumentInput() ? nsamples : max_batch_size_);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        auto alpha = alpha_[sample_idx].data[0];
        DALI_ENFORCE(alpha > 0 && std::isfinite(alpha),
                     make_string("The `alpha` must be a positive float32, got `", alpha,
                                 "` for sample at index ", sample_idx, "."));
      }
    }
    if (beta_.HasArgumentInput() || !beta_.size()) {
      beta_.Acquire(spec, ws, beta_.HasArgumentInput() ? nsamples : max_batch_size_);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        auto beta = beta_[sample_idx].data[0];
        DALI_ENFORCE(beta > 0 && std::isfinite(beta),
                     make_string("The `beta` must be a positive float32, got `", beta,
                                 "` for sample at index ", sample_idx, "."));
      }
    }
  }

  DALIDataType DefaultDataType(const OpSpec &spec, const Workspace &ws) const {
    return DALI_FLOAT;
  }

  template <typename T>
  bool SetupDists(BetaDistributionImpl<T> *dists, const Workspace &ws, int nsamples) {
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto alpha = alpha_[sample_idx].data[0];
      auto beta = beta_[sample_idx].data[0];
      dists[sample_idx] =
          BetaDistributionImpl<T>{alpha_[sample_idx].data[0], beta_[sample_idx].data[0]};
    }
    return true;
  }

  template <typename T>
  void RunImplTyped(Workspace &ws) {
    Base::template RunImplTyped<T, BetaDistributionImpl<T>>(ws);
  }

  void RunImpl(Workspace &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, (DALI_BETA_DIST_TYPES), (
      this->template RunImplTyped<T>(ws);
    ), (  // NOLINT
      DALI_FAIL(make_string("Data type ", dtype_, " is not supported. "
                            "Supported types are : ", ListTypeNames<DALI_BETA_DIST_TYPES>()));
    ));  // NOLINT
  }

 protected:
  using Base::dtype_;
  using Base::max_batch_size_;

  ArgValue<float, 0> alpha_;
  ArgValue<float, 0> beta_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_BETA_DISTRIBUTION_H_
