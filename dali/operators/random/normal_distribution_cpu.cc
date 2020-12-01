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

#include "dali/operators/random/rng_base_cpu.h"
#include "dali/pipeline/operator/arg_helper.h"

#define DALI_NORMDIST_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, \
                             int64_t, float16, float, double)

namespace dali {

DALI_SCHEMA(NewNormalDistribution)
    .DocStr(R"code(Generates random numbers following a normal distribution.

The shape of the generated data can be either specified explicitly with a ``shape`` argument,
or chosen to match the shape of the input, if provided. If none are present, a single number is
generated.
)code")
    .NumInput(0, 1)
    .NumOutput(1)
    .AddOptionalArg("mean", R"code(Mean of the distribution.)code",
                    0.f, true)
    .AddOptionalArg("stddev",
                    R"code(Standard deviation of the distribution.)code",
                    1.f, true)
    .AddParent("RNGAttr");


class NormalDistributionCPU : public RNGBaseCPU<NormalDistributionCPU> {
 public:
  explicit NormalDistributionCPU(const OpSpec &spec)
      : RNGBaseCPU<NormalDistributionCPU>(spec), mean_("mean", spec), stddev_("stddev", spec) {}

  ~NormalDistributionCPU() override = default;

  void SetupImplImpl(const OpSpec &spec, const workspace_t<CPUBackend> &ws, int nsamples) {
    mean_.Acquire(spec, ws, nsamples);
    stddev_.Acquire(spec, ws, nsamples);
    dist_.clear();
    dist_.reserve(nsamples);
    for (int s = 0; s < nsamples; s++) {
      dist_.emplace_back(mean_[s].data[0], stddev_[s].data[0]);
    }
  }

  template <typename T>
  void Generate(span<T> out, int sample_idx, std::mt19937_64& rng) {
    auto &dist = dist_[sample_idx];
    for (auto &value : out)
      value = ConvertSat<T>(dist(rng));
  }

  void RunImpl(workspace_t<CPUBackend> &ws) override {
    TYPE_SWITCH(dtype_, type2id, T, DALI_NORMDIST_TYPES, (
      RunImplTyped<T>(ws);
    ), DALI_FAIL(make_string("Unsupported data type: ", dtype_)));  // NOLINT
  }

  DALIDataType DefaultDataType() const {
    return DALI_FLOAT;
  }

 private:
  ArgValue<float> mean_;
  ArgValue<float> stddev_;
  std::vector<std::normal_distribution<float>> dist_;
};

DALI_REGISTER_OPERATOR(NewNormalDistribution, NormalDistributionCPU, CPU);

}  // namespace dali
