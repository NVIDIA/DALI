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

template <bool NoiseGen>
struct RNGBaseFields<CPUBackend, NoiseGen> {
  RNGBaseFields(int64_t seed, int nsamples) {}

  std::vector<uint8_t> dists_cpu_;

  void ReserveDistsData(size_t nbytes) {
    dists_cpu_.reserve(nbytes);
  }
};

template <bool NoiseGen>
struct DistGen;

template <>
struct DistGen<false> {
  template <typename T, typename Dist, typename RNG>
  inline void gen(span<T> out, span<const T> in, Dist& dist, RNG &rng) const {
    (void) in;
    for (auto &x : out)
      x = ConvertSat<T>(dist.Generate(rng));
  }
};

template <>
struct DistGen<true> {
  template <typename T, typename Dist, typename RNG>
  inline void gen(span<T> out, span<const T> in, Dist& dist, RNG &rng) const {
    assert(out.size() == in.size());
    for (int64_t k = 0; k < out.size(); k++) {
      auto n = dist.Generate(in[k], rng);
      dist.Apply(out[k], in[k], n);
    }
  }
};

template <typename T>
inline span<T> get_chunk(span<T> data, int c, int chunks) {
  T* start = data.begin() + data.size() * c / chunks;
  T* end = data.begin() + data.size() * (c + 1) / chunks;
  return make_span(start, end - start);
}

template <typename Backend, typename Impl, bool NoiseGen>
template <typename T, typename Dist>
void RNGBase<Backend, Impl, NoiseGen>::RunImplTyped(workspace_t<CPUBackend> &ws) {
  // Should never be called for Backend != CPUBackend
  static_assert(std::is_same<Backend, CPUBackend>::value, "Invalid backend");
  auto &output = ws.template OutputRef<CPUBackend>(0);
  auto out_view = view<T>(output);
  const auto &out_shape = out_view.shape;
  auto &tp = ws.GetThreadPool();
  constexpr int64_t kThreshold = 1 << 18;
  constexpr int64_t kChunkSize = 1 << 16;
  constexpr size_t kNumChunkSeeds = 16;
  int nsamples = output.shape().size();

  TensorListView<detail::storage_tag_map_t<Backend>, const T, DynamicDimensions> in_view;
  if (NoiseGen) {
    const auto &input = ws.InputRef<CPUBackend>(0);
    in_view = view<const T>(input);
    output.SetLayout(input.GetLayout());
  }

  auto &dists_cpu = backend_data_.dists_cpu_;
  dists_cpu.resize(sizeof(Dist) * nsamples);  // memory was already reserved in the constructor
  Dist* dists = reinterpret_cast<Dist*>(dists_cpu.data());
  bool use_default_dist = !This().template SetupDists<T>(dists, nsamples);

  DistGen<NoiseGen> dist_gen_;
  for (int sample_id = 0; sample_id < nsamples; ++sample_id) {
    auto sample_sz = out_shape.tensor_size(sample_id);
    span<T> out_span{out_view[sample_id].data, sample_sz};
    span<const T> in_span;
    if (NoiseGen) {
      assert(sample_sz == in_view.shape.tensor_size(sample_id));
      in_span = {in_view[sample_id].data, sample_sz};
    }
    if (sample_sz < kThreshold) {
      tp.AddWork(
        [=](int thread_id) {
          auto dist = use_default_dist ? Dist() : dists[sample_id];
          dist_gen_.template gen<T>(out_span, in_span, dist, rng_[sample_id]);
        }, sample_sz);
    } else {
      int chunks = div_ceil(out_span.size(), kChunkSize);
      std::array<uint32_t, kNumChunkSeeds> seed;
      for (int c = 0; c < chunks; c++) {
        auto out_chunk = get_chunk<T>(out_span, c, chunks);
        span<const T> in_chunk;
        if (NoiseGen) {
          in_chunk = get_chunk<const T>(in_span, c, chunks);
          assert(out_chunk.size() == in_chunk.size());
        }
        for (auto &s : seed)
          s = rng_[sample_id]();
        tp.AddWork(
          [=](int thread_id) {
            std::seed_seq seq(seed.begin(), seed.end());
            std::mt19937_64 chunk_rng(seq);
            auto dist = use_default_dist ? Dist() : dists[sample_id];
            dist_gen_.template gen<T>(out_chunk, in_chunk, dist, chunk_rng);
          }, out_chunk.size());
      }
    }
  }
  tp.RunAll();
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
