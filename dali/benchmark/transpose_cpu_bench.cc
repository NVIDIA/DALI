// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <benchmark/benchmark.h>
#include <numeric>
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/mm/memory.h"

namespace dali {

namespace {

using CaseData = std::tuple<TensorShape<>, std::vector<int>>;

static CaseData cases[] = {
    // HWC small cases
    CaseData{{8, 6, 3}, {0, 1, 2}},           // HWC
    CaseData{{8, 6, 3}, {2, 0, 1}},           // CHW
    CaseData{{8, 6, 3}, {2, 1, 0}},           // CWH
                                              // HWC small cases
    CaseData{{8, 6, 4}, {0, 1, 2}},           // HWC
    CaseData{{8, 6, 4}, {2, 0, 1}},           // CHW
    CaseData{{8, 6, 4}, {2, 1, 0}},           // CWH
                                              // HWC bigger cases
    CaseData{{100, 60, 3}, {0, 1, 2}},        // HWC
    CaseData{{100, 60, 3}, {2, 0, 1}},        // CHW
    CaseData{{100, 60, 3}, {2, 1, 0}},        // CWH
                                              // 4D
    CaseData{{20, 20, 20, 4}, {0, 1, 2, 3}},  // id
    CaseData{{20, 20, 20, 4}, {3, 2, 1, 0}},
    CaseData{{20, 20, 20, 4}, {0, 1, 3, 2}},
    CaseData{{20, 20, 20, 4}, {0, 3, 1, 2}},
    // 8D
    CaseData{{7, 2, 4, 6, 10, 8, 4, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
    CaseData{{7, 2, 4, 6, 10, 8, 4, 2}, {6, 4, 2, 0, 7, 5, 3, 1}},
    CaseData{{7, 2, 4, 6, 10, 8, 4, 2}, {5, 4, 2, 1, 7, 6, 3, 0}},
    CaseData{{7, 2, 4, 6, 10, 8, 4, 2}, {7, 5, 3, 2, 4, 0, 1, 6}},
};

std::tuple<TensorShape<>, std::vector<int>> GetCase(int id) {
  return cases[id];
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (unsigned int i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
    for (int scale = 1; scale <= 8; scale *= 2) {
      b->Args({i, scale});
    }
  }
}

}  // namespace

template <typename T>
class TransposeFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& st) override {
    auto test_case = GetCase(st.range(0));
    src_shape_ = std::get<0>(test_case);
    int dim = 0, total_dims = src_shape_.size();
    for (auto& elem : src_shape_) {
      if (dim++ < 4) elem *= st.range(1);
    }
    perm_ = std::get<1>(test_case);
    dst_shape_ = permute(src_shape_, perm_);
    auto total_size = volume(src_shape_);
    dst_mem_.resize(total_size);
    src_mem_.resize(total_size);
    for (int64_t i = 0; i < total_size; i++) {
      src_mem_[i] = i;
    }
    src_view_ = TensorView<StorageCPU, const T>(src_mem_.data(), src_shape_);
    dst_view_ = TensorView<StorageCPU, T>(dst_mem_.data(), dst_shape_);
  }

  void TearDown(benchmark::State& st) override {
    dst_mem_.clear();
    dst_mem_.shrink_to_fit();
    src_mem_.clear();
    src_mem_.shrink_to_fit();
  }

  using TransposeFunction = void(const TensorView<StorageCPU, T>&,
                                 const TensorView<StorageCPU, const T>&, span<const int>);

  template <TransposeFunction F>
  void benchmark() {
    benchmark::DoNotOptimize(src_mem_.data());
    F(dst_view_, src_view_, make_span(perm_));
    benchmark::DoNotOptimize(dst_mem_.data());
    benchmark::ClobberMemory();
  }

  std::vector<int> perm_;

  TensorView<StorageCPU, T> dst_view_;
  TensorView<StorageCPU, const T> src_view_;
  TensorShape<> src_shape_, dst_shape_;
  std::vector<T> dst_mem_, src_mem_;
};

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, Uint8Test, uint8_t)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::Transpose<uint8_t>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, Uint16Test, uint16_t)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::Transpose<uint16_t>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, IntTest, int)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::Transpose<int>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, DoubleTest, double)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::Transpose<double>>();
  }
}

BENCHMARK_REGISTER_F(TransposeFixture, Uint8Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, Uint16Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, IntTest)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, DoubleTest)->Apply(CustomArguments);

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, CompactUint8Test, uint8_t)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::TransposeGrouped<uint8_t>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, CompactUint16Test, uint16_t)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::TransposeGrouped<uint16_t>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, CompactIntTest, int)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::TransposeGrouped<int>>();
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TransposeFixture, CompactDoubleTest, double)(benchmark::State& st) {
  for (auto _ : st) {
    benchmark<&kernels::TransposeGrouped<double>>();
  }
}

BENCHMARK_REGISTER_F(TransposeFixture, CompactUint8Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, CompactUint16Test)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, CompactIntTest)->Apply(CustomArguments);
BENCHMARK_REGISTER_F(TransposeFixture, CompactDoubleTest)->Apply(CustomArguments);

}  // namespace dali
