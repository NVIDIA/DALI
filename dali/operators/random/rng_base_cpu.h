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

#ifndef DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
#define DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_

#include <random>
#include <vector>
#include "dali/operators/random/rng_base.h"
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

template <>
struct RNGBaseFields<CPUBackend> {
  RNGBaseFields(int64_t seed, int nsamples) {}

  std::vector<uint8_t> dists_cpu_;

  void ReserveDistsData(size_t nbytes) {
    dists_cpu_.reserve(nbytes);
  }
};

template <typename Backend, typename Impl>
template <typename T, typename Dist>
void RNGBase<Backend, Impl>::RunImplTyped(workspace_t<CPUBackend> &ws) {
  // Should never be called for Backend != CPUBackend
  static_assert(std::is_same<Backend, CPUBackend>::value, "Invalid backend");
  auto &output = ws.OutputRef<CPUBackend>(0);
  auto out_view = view<T>(output);
  const auto &out_shape = out_view.shape;
  auto &tp = ws.GetThreadPool();
  constexpr int64_t kThreshold = 1 << 18;
  constexpr int64_t kChunkSize = 1 << 16;
  constexpr size_t kNumChunkSeeds = 16;
  int nsamples = output.shape().size();

  auto &dists_cpu = backend_data_.dists_cpu_;
  dists_cpu.resize(sizeof(Dist) * nsamples);  // memory was already reserved in the constructor

  Dist* dists = reinterpret_cast<Dist*>(dists_cpu.data());
  bool use_default_dist = !This().template SetupDists<Dist>(dists, nsamples);

  for (int sample_id = 0; sample_id < nsamples; ++sample_id) {
    auto sample_sz = out_shape.tensor_size(sample_id);
    span<T> out_span{out_view[sample_id].data, sample_sz};
    if (sample_sz < kThreshold) {
      tp.AddWork(
        [=](int thread_id) {
          auto dist = use_default_dist ? Dist() : dists[sample_id];
          for (auto &x : out_span)
            x = ConvertSat<T>(dist(rng_[sample_id]));
        }, sample_sz);
    } else {
      int chunks = div_ceil(out_span.size(), kChunkSize);
      std::array<uint32_t, kNumChunkSeeds> seed;
      for (int c = 0; c < chunks; c++) {
        auto start = out_span.begin() + out_span.size() * c / chunks;
        auto end = out_span.begin() + out_span.size() * (c + 1) / chunks;
        auto chunk = make_span(start, end - start);
        for (auto &s : seed)
          s = rng_[sample_id]();
        tp.AddWork(
          [=](int thread_id) {
            std::seed_seq seq(seed.begin(), seed.end());
            std::mt19937_64 chunk_rng(seq);
            auto dist = use_default_dist ? Dist() : dists[sample_id];
            for (auto &x : chunk)
              x = ConvertSat<T>(dist(chunk_rng));
          }, chunk.size());
      }
    }
    tp.RunAll();
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
