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
#include <utility>
#include <vector>
#include "dali/operators/random/rng_base.h"
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"

namespace dali {

template <bool IsNoiseGen>
struct RNGBaseFields<CPUBackend, IsNoiseGen> {
  RNGBaseFields(int64_t seed, int nsamples) {}

  std::vector<uint8_t> dists_cpu_;

  void ReserveDistsData(size_t nbytes) {
    dists_cpu_.reserve(nbytes);
  }
};

template <bool IsNoiseGen>
struct DistGen;

template <>
struct DistGen<false> {
  template <typename T, typename Dist, typename RNG>
  inline void gen(span<T> out, span<const T> in, Dist &dist, RNG &rng,
                  int64_t p_offset, int64_t p_count) const {
    (void) in;
    int64_t p_pos = p_offset;
    for (int64_t p = 0; p < p_count; p++, p_pos++) {
      out[p_pos] = ConvertSat<T>(dist.Generate(rng));
    }
  }

  template <typename T, typename Dist, typename RNG>
  inline void gen_all_channels(span<T> out, span<const T> in, Dist &dist, RNG &rng,
                               int64_t p_offset, int64_t p_count, int c_count,
                               int64_t c_stride, int64_t p_stride) const {
    (void) in;
    int64_t p_pos = p_offset * p_stride;
    for (int64_t p = 0; p < p_count; p++, p_pos += p_stride) {
      int64_t c_pos = p_pos;
      auto n = ConvertSat<T>(dist.Generate(rng));
      for (int c = 0; c < c_count; c++, c_pos += c_stride) {
        out[c_pos] = n;
      }
    }
  }
};

template <>
struct DistGen<true> {
  template <typename T, typename Dist, typename RNG>
  inline void gen(span<T> out, span<const T> in, Dist& dist, RNG &rng,
                  int64_t p_offset, int64_t p_count) const {
    assert(out.size() == in.size());
    int64_t p_pos = p_offset;
    for (int64_t p = 0; p < p_count; p++, p_pos++) {
      auto n = dist.Generate(in[p_pos], rng);
      dist.Apply(out[p_pos], in[p_pos], n);
    }
  }

  template <typename T, typename Dist, typename RNG>
  inline void gen_all_channels(span<T> out, span<const T> in, Dist& dist, RNG &rng,
                               int64_t p_offset, int64_t p_count,
                               int c_count, int64_t c_stride, int64_t p_stride) const {
    assert(out.size() == in.size());
    int64_t p_pos = p_offset * p_stride;
    for (int64_t p = 0; p < p_count; p++, p_pos += p_stride) {
      int64_t c_pos = p_pos;
      auto n = dist.Generate(in[p_pos], rng);
      for (int c = 0; c < c_count; c++, c_pos += c_stride) {
        dist.Apply(out[c_pos], in[c_pos], n);
      }
    }
  }
};

template <typename T>
inline std::pair<int64_t, int64_t> get_chunk(int64_t npixels, int c, int chunks) {
  int64_t start = npixels * c / chunks;
  int64_t end = npixels * (c + 1) / chunks;
  return {start, end - start};
}

template <typename Backend, typename Impl, bool IsNoiseGen>
template <typename T, typename Dist>
void RNGBase<Backend, Impl, IsNoiseGen>::RunImplTyped(workspace_t<CPUBackend> &ws) {
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
  int ndim = output.shape().sample_dim();

  TensorListView<detail::storage_tag_map_t<Backend>, const T, DynamicDimensions> in_view;
  if (IsNoiseGen) {
    const auto &input = ws.InputRef<CPUBackend>(0);
    in_view = view<const T>(input);
    output.SetLayout(input.GetLayout());
  }

  // TODO(janton): set layout explicitly from the user for RNG

  auto &dists_cpu = backend_data_.dists_cpu_;
  dists_cpu.resize(sizeof(Dist) * nsamples);  // memory was already reserved in the constructor
  Dist* dists = reinterpret_cast<Dist*>(dists_cpu.data());
  bool use_default_dist = !This().template SetupDists<T>(dists, nsamples);

  int channel_dim = -1;
  auto layout = output.GetLayout();
  channel_dim = layout.empty() ? ndim - 1 : layout.find('C');

  DistGen<IsNoiseGen> dist_gen_;
  for (int sample_id = 0; sample_id < nsamples; ++sample_id) {
    auto sample_sz = out_shape.tensor_size(sample_id);
    int64_t total_p_count = sample_sz;
    int nchannels = -1;
    span<T> out_span{out_view[sample_id].data, sample_sz};
    span<const T> in_span;
    if (IsNoiseGen) {
      assert(sample_sz == in_view.shape.tensor_size(sample_id));
      in_span = {in_view[sample_id].data, sample_sz};
    }

    int64_t c_stride = -1, p_stride = -1;
    bool independent_channels = This().PerChannel();
    if (!independent_channels) {
      DALI_ENFORCE(
          channel_dim == 0 || channel_dim == ndim - 1,
          make_string("'C' should be the first or the last dimension in the layout, "
                      "except for empty layouts where channel-last is assumed. "
                      "Got layout: \"", layout, "\"."));
      auto sh = out_shape.tensor_shape_span(sample_id);
      nchannels = sh[channel_dim];
      total_p_count /= nchannels;
      c_stride = volume(sh.begin() + channel_dim + 1, sh.end());
      p_stride = channel_dim == 0 ? 1 : nchannels;
    }

    if (total_p_count < kThreshold) {
      tp.AddWork(
        [=](int thread_id) {
          auto dist = use_default_dist ? Dist() : dists[sample_id];
          if (independent_channels) {
            dist_gen_.template gen<T>(out_span, in_span, dist, rng_[sample_id], 0, total_p_count);
          } else {
            dist_gen_.template gen_all_channels<T>(out_span, in_span, dist, rng_[sample_id], 0,
                                                   total_p_count, nchannels, c_stride, p_stride);
          }
        }, total_p_count);
    } else {
      int chunks = div_ceil(total_p_count, kChunkSize);
      std::array<uint32_t, kNumChunkSeeds> seed;
      for (int c = 0; c < chunks; c++) {
        int64_t p_offset, p_count;
        std::tie(p_offset, p_count) = get_chunk<T>(total_p_count, c, chunks);
        for (auto &s : seed)
          s = rng_[sample_id]();
        tp.AddWork(
          [=](int thread_id) {
            std::seed_seq seq(seed.begin(), seed.end());
            std::mt19937_64 chunk_rng(seq);
            auto dist = use_default_dist ? Dist() : dists[sample_id];
            if (independent_channels) {
              dist_gen_.template gen<T>(out_span, in_span, dist, chunk_rng,
                                        p_offset, p_count);
            } else {
              dist_gen_.template gen_all_channels<T>(out_span, in_span, dist, chunk_rng, p_offset,
                                                     p_count, nchannels, c_stride, p_stride);
            }
          }, p_count);
      }
    }
  }
  tp.RunAll();
}

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_CPU_H_
