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

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/erase/erase_gpu.h"
#include "dali/kernels/scratch.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

#include "dali/kernels/erase/erase_cpu.h"

#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {

template <int ndim>
void debug_print(TensorListView<StorageCPU, uint8_t, ndim> tlv, int height, int width) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      std::cout << std::setw(2) << int(*tlv[0](y, x)) % 100;
    }
    std::cout << endl;
  }
}

void verify_regions(ivec<4> region_shape, ivec<4> sample_shape, ivec<4> expected_cover) {
  auto cover = div_ceil(sample_shape, region_shape);
  EXPECT_EQ(cover, expected_cover);
  int idx = 0;
  for (int d0 = 0; d0 < expected_cover[0]; d0++) {
    for (int d1 = 0; d1 < expected_cover[1]; d1++) {
      for (int d2 = 0; d2 < expected_cover[2]; d2++) {
        for (int d3 = 0; d3 < expected_cover[3]; d3++) {
          auto regions_start = get_region_start(idx, region_shape, sample_shape);
          ivec<4> region_start_expected = {d0 * region_shape[0], d1 * region_shape[1],
                                           d2 * region_shape[2], d3 * region_shape[3]};
          EXPECT_EQ(regions_start, region_start_expected);
          idx++;
        }
      }
    }
  }
}

TEST(EraseGpuKernelTest, CheckUtils) {
  {
    ivec<4> region_shape = {2, 2, 32, 32};
    ivec<4> sample_shape = {16, 8, 64, 64};
    ivec<4> expected_cover = {8, 4, 2, 2};
    verify_regions(region_shape, sample_shape, expected_cover);
  }
  {
    ivec<4> region_shape = {2, 2, 32, 32};
    ivec<4> sample_shape = {16, 8, 32, 64};
    ivec<4> expected_cover = {8, 4, 1, 2};
    verify_regions(region_shape, sample_shape, expected_cover);
  }
  {
    ivec<4> region_shape = {2, 2, 32, 32};
    ivec<4> sample_shape = {16, 8, 64, 32};
    ivec<4> expected_cover = {8, 4, 2, 1};
    verify_regions(region_shape, sample_shape, expected_cover);
  }
  {
    ivec<4> region_shape = {2, 2, 32, 32};
    ivec<4> sample_shape = {16, 8, 32, 32};
    ivec<4> expected_cover = {8, 4, 1, 1};
    verify_regions(region_shape, sample_shape, expected_cover);
  }
}

template <int ndim>
struct EraseTestParams {
  int num_erase_regions;
  int region_generation;
  int fill_type;
  TensorShape<ndim> shape;
};

template <int ndim>
std::ostream& operator<<(std::ostream& os, const EraseTestParams<ndim>& p) {
  os << "Num erase regions: " << p.num_erase_regions << ", region generation: "
     << p.region_generation << ", fill type: " << p.fill_type << ", shape: " << p.shape;
  return os;
}

template <typename T, int ndim, int channel_dim = -1>
struct EraseGpuKernelTest :
    public testing::TestWithParam<EraseTestParams<ndim>> {
  void SetUp() override {
    auto params = this->GetParam();
    num_erase_regions_ = params.num_erase_regions;
    region_generation_ = params.region_generation;
    fill_type_ = params.fill_type;
    shape_ = params.shape;
    test_shape_ = uniform_list_shape<ndim>(batch_size_, shape_);

    input_.reshape(test_shape_);
    output_.reshape(test_shape_);
    baseline_.reshape(test_shape_);
    auto cpu_input_view = input_.cpu();
    SequentialFill(cpu_input_view);
    if (fill_type_ == 2) {
      fill_values_.resize(0);
    } else if (fill_type_ == 1) {
      fill_values_.resize(shape_[channel_dim]);
      int value = 0;
      for (auto &elem : fill_values_) {
        elem = value++;
      }
    } else if (fill_type_ == 0) {
      fill_values_.resize(1);
      fill_values_[0] = 42;
    }
  }

  void RunTest() {
    if (region_generation_ == 0) {
      std::cerr << ">> No cover" << std::endl;
    } else if (region_generation_ == 1) {
      std::cerr << ">> Full cover" << std::endl;
    } else if (region_generation_ == 2) {
      std::cerr << ">> Random cover of size: " << num_erase_regions_ << std::endl;
    }
    EraseGpu<T, ndim, channel_dim> kernel;
    KernelContext ctx;
    ctx.gpu.stream = 0;

    CreateRegions();

    auto regions_gpu = regions_.gpu();

    auto in_view = input_.gpu();

    auto req = kernel.Setup(ctx, in_view, regions_gpu, make_span(fill_values_));

    auto out_view = output_.gpu();

    ScratchpadAllocator scratch_alloc;
    scratch_alloc.Reserve(req.scratch_sizes);
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;

    kernel.Run(ctx, out_view, in_view, regions_gpu, make_span(fill_values_));

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaGetLastError());

    RepackAndCalcCpu();
    Verify();
  }

  void Verify() {
    auto cpu_out_view = output_.cpu();
    auto cpu_baseline_view = baseline_.cpu();
    Check(cpu_out_view, cpu_baseline_view);
  }

  void RepackAndCalcCpu() {
    auto input_tlv = input_.cpu();
    auto baseline_tlv = baseline_.cpu();
    auto regions_tlv = regions_.cpu();
    for (int i = 0; i < batch_size_; i++) {
      auto baseline_tv = baseline_tlv[i];
      auto input_tv = input_tlv[i];

      EraseArgs<T, ndim> args;
      args.rois.resize(num_erase_regions_);
      for (int j = 0; j < num_erase_regions_; j++) {
        for (int d = 0; d < ndim; d++) {
          args.rois[j].anchor[d] = regions_tlv[i](j)->lo[d];
          args.rois[j].shape[d] = regions_tlv[i](j)->hi[d] - regions_tlv[i](j)->lo[d];
          args.rois[j].fill_values = fill_values_;
          args.rois[j].channels_dim = channel_dim;
        }
      }

      EraseCpu<T, ndim> cpu_kernel;
      KernelContext ctx;
      cpu_kernel.Run(ctx, baseline_tv, input_tv, args);
    }
  }

  void CreateRegions() {
    if (region_generation_ == 0) {
      num_erase_regions_ = 0;
      // no cover
    } else if (region_generation_ == 1) {
      // full cover
      num_erase_regions_ = 1;
    }
    TensorListShape<1> regions_shape = uniform_list_shape<1>(batch_size_, {num_erase_regions_});
    regions_.reshape(regions_shape);
    auto regions_cpu = regions_.cpu();
    if (region_generation_ == 1) {
      // full cover
      for (int i = 0; i < batch_size_; i++) {
        auto regions_tv = regions_cpu[i];
        *regions_tv(0) = ibox<ndim>({0}, to_ivec(shape_));
      }
    } else if (region_generation_ == 2) {
      std::mt19937 gen(0);
      for (int i = 0; i < batch_size_; i++) {
        auto regions_tv = regions_cpu[i];
        for (int j = 0; j < num_erase_regions_; j ++) {
          ibox<ndim> region_box;
          for (int d = 0; d < ndim; d++) {
            std::uniform_int_distribution<>  start_dim(0, shape_[d] - 1);
            region_box.lo[d] = start_dim(gen);
            std::uniform_int_distribution<>  end_dim(region_box.lo[d] + 1, shape_[d]);
            region_box.hi[d] = end_dim(gen);
          }
          *regions_tv(j) = region_box;
        }
      }
    }
  }

  int num_erase_regions_;
  int region_generation_;
  bool fill_type_;
  std::vector<T> fill_values_;
  TensorShape<ndim> shape_;
  TensorListShape<ndim> test_shape_;
  constexpr static int batch_size_ = 16;
  TestTensorList<T, ndim> input_, output_, baseline_;
  TestTensorList<ibox<ndim>, 1> regions_;
};

using EraseGpuKernel1fTest = EraseGpuKernelTest<float, 1>;
using EraseGpuKernel2fTest = EraseGpuKernelTest<float, 2>;
using EraseGpuKernel2NCfTest = EraseGpuKernelTest<float, 2, 1>;
using EraseGpuKernel3fTest = EraseGpuKernelTest<float, 3>;
using EraseGpuKernel3fHWCTest = EraseGpuKernelTest<float, 3, 2>;
using EraseGpuKernel3fCHWTest = EraseGpuKernelTest<float, 3, 0>;
using EraseGpuKernel4fDHCWTest = EraseGpuKernelTest<float, 4, 2>;
using EraseGpuKernel4fDHWCTest = EraseGpuKernelTest<float, 4, 3>;
using EraseGpuKernel5fTest = EraseGpuKernelTest<float, 5>;

#define ERASE_TEST_P(TEST) \
  TEST_P(TEST, RunAndVerify) { \
    this->RunTest(); \
  }

ERASE_TEST_P(EraseGpuKernel1fTest)
ERASE_TEST_P(EraseGpuKernel2fTest)
ERASE_TEST_P(EraseGpuKernel2NCfTest)
ERASE_TEST_P(EraseGpuKernel3fTest)
ERASE_TEST_P(EraseGpuKernel3fHWCTest)
ERASE_TEST_P(EraseGpuKernel3fCHWTest)
ERASE_TEST_P(EraseGpuKernel4fDHCWTest)
ERASE_TEST_P(EraseGpuKernel4fDHWCTest)
ERASE_TEST_P(EraseGpuKernel5fTest)

// Parameters for tests are:
// <number of erase regions>, <generation scheme>, <fill_type>, <shape>
// generation scheme:
// * 0 - only copy, no erase
// * 1 - full, 1-element cover, only erase
// * 2 - random cover
// fill_type:
// * 0 - one element `42` fill
// * 1 - per channel, consecutive values
// * 2 - default `0` fill

std::vector<EraseTestParams<1>> values_1 = {
    {0, 0, 0, {512 * 1024}},
    {1, 1, 0, {512 * 1024}},
    {1, 2, 0, {512 * 1024}},
    {10, 2, 0, {512 * 1024}},
    {100, 2, 0, {512 * 1024}},
    {0, 0, 2, {512 * 1024}},
    {1, 1, 2, {512 * 1024}},
    {1, 2, 2, {512 * 1024}},
    {10, 2, 2, {512 * 1024}},
    {100, 2, 2, {512 * 1024}},
};

std::vector<EraseTestParams<2>> values_2 = {
    {0, 0, 0, {512, 1024}},
    {1, 1, 0, {512, 1024}},
    {1, 2, 0, {512, 1024}},
    {10, 2, 0, {512, 1024}},
    {100, 2, 0, {512, 1024}},
};

std::vector<EraseTestParams<2>> values_2NC = {
    {0, 0, 0, {512 * 1024, 3}},
    {1, 1, 0, {512 * 1024, 3}},
    {1, 2, 0, {512 * 1024, 3}},
    {10, 2, 1, {512 * 1024, 3}},
    {100, 2, 2, {512 * 1024, 3}},
};

std::vector<EraseTestParams<3>> values_3 = {
    {0, 0, 0, {256, 256, 256}},
    {1, 1, 0, {256, 256, 256}},
    {1, 2, 0, {256, 256, 256}},
    {10, 2, 0, {256, 256, 256}},
    {100, 2, 0, {256, 256, 256}},
    {1000, 2, 0, {256, 256, 256}},
};

std::vector<EraseTestParams<3>> values_3HWC = {
    {0, 0, 0, {256, 256, 1}},    {0, 0, 0, {256, 256, 3}},    {0, 0, 0, {256, 256, 4}},
    {0, 0, 0, {256, 256, 8}},    {0, 0, 0, {256, 256, 16}},   {0, 0, 0, {256, 256, 64}},
    {1, 1, 0, {256, 256, 1}},    {1, 1, 0, {256, 256, 3}},    {1, 1, 0, {256, 256, 4}},
    {1, 1, 0, {256, 256, 8}},    {1, 1, 0, {256, 256, 16}},   {1, 1, 0, {256, 256, 64}},

    {0, 0, 1, {256, 256, 1}},    {0, 0, 1, {256, 256, 3}},    {0, 0, 1, {256, 256, 4}},
    {0, 0, 1, {256, 256, 8}},    {0, 0, 1, {256, 256, 16}},   {0, 0, 1, {256, 256, 64}},
    {1, 1, 1, {256, 256, 1}},    {1, 1, 1, {256, 256, 3}},    {1, 1, 1, {256, 256, 4}},
    {1, 1, 1, {256, 256, 8}},    {1, 1, 1, {256, 256, 16}},   {1, 1, 1, {256, 256, 64}},

    {0, 0, 2, {256, 256, 1}},    {0, 0, 2, {256, 256, 3}},    {0, 0, 2, {256, 256, 4}},
    {0, 0, 2, {256, 256, 8}},    {0, 0, 2, {256, 256, 16}},   {0, 0, 2, {256, 256, 64}},
    {1, 1, 2, {256, 256, 1}},    {1, 1, 2, {256, 256, 3}},    {1, 1, 2, {256, 256, 4}},
    {1, 1, 2, {256, 256, 8}},    {1, 1, 2, {256, 256, 16}},   {1, 1, 2, {256, 256, 64}},

    {1, 2, 1, {256, 256, 1}},    {10, 2, 1, {256, 256, 1}},   {100, 2, 1, {256, 256, 1}},
    {1000, 2, 1, {256, 256, 1}},
    {1, 2, 1, {256, 256, 3}},    {10, 2, 1, {256, 256, 3}},   {100, 2, 1, {256, 256, 3}},
    {1000, 2, 1, {256, 256, 3}},
    {1, 2, 1, {256, 256, 16}},    {10, 2, 1, {256, 256, 16}}, {100, 2, 1, {256, 256, 16}},
    {1000, 2, 1, {256, 256, 16}},
};

std::vector<EraseTestParams<3>> values_3CHW = {
    {0, 0, 0, {3, 256, 256}},
    {1, 1, 0, {3, 256, 256}},
    {0, 0, 0, {16, 256, 256}},
    {1, 1, 0, {16, 256, 256}},
    {1, 2, 0, {3, 256, 256}},
    {10, 2, 0, {3, 256, 256}},
    {100, 2, 0, {3, 256, 256}},
    {1000, 2, 0, {3, 256, 256}},
};

std::vector<EraseTestParams<4>> values_4DHCW = {
    {0, 0, 0, {64, 64, 3, 64}},     {0, 0, 0, {64, 64, 4, 64}},    {0, 0, 0, {64, 64, 8, 64}},
    {1, 1, 0, {64, 64, 3, 64}},     {1, 1, 0, {64, 64, 4, 64}},    {1, 1, 0, {64, 64, 8, 64}},
    {0, 0, 1, {64, 64, 3, 64}},     {0, 0, 1, {64, 64, 4, 64}},    {0, 0, 1, {64, 64, 8, 64}},
    {1, 1, 1, {64, 64, 3, 64}},     {1, 1, 1, {64, 64, 4, 64}},    {1, 1, 1, {64, 64, 8, 64}},
    {1, 2, 1, {64, 64, 3, 64}},     {10, 2, 1, {64, 64, 3, 64}},   {100, 2, 1, {64, 64, 3, 64}},
    {100, 2, 1, {64, 256, 3, 256}}, {1000, 2, 1, {64, 64, 3, 64}}, {1000, 2, 1, {64, 256, 3, 256}},
};

std::vector<EraseTestParams<4>> values_4DHWC = {
    {0, 0, 0, {64, 64, 64, 3}},     {0, 0, 0, {64, 64, 64, 4}},    {0, 0, 0, {64, 64, 64, 8}},
    {1, 1, 0, {64, 64, 64, 3}},     {1, 1, 0, {64, 64, 64, 4}},    {1, 1, 0, {64, 64, 64, 8}},
    {0, 0, 1, {64, 64, 64, 3}},     {0, 0, 1, {64, 64, 64, 4}},    {0, 0, 1, {64, 64, 64, 8}},
    {1, 1, 1, {64, 64, 64, 3}},     {1, 1, 1, {64, 64, 64, 4}},    {1, 1, 1, {64, 64, 64, 8}},
    {1, 2, 1, {64, 64, 64, 3}},     {10, 2, 1, {64, 64, 64, 3}},   {100, 2, 1, {64, 64, 64, 3}},
    {100, 2, 1, {64, 256, 256, 3}}, {1000, 2, 1, {64, 64, 64, 3}}, {1000, 2, 1, {64, 256, 256, 3}},
};

std::vector<EraseTestParams<5>> values_5 = {
    {0, 0, 0, {4, 6, 64, 64, 64}},     {1, 1, 0, {4, 6, 64, 64, 64}},
    {0, 0, 0, {2, 32, 16, 256, 256}},  {1, 1, 0, {2, 4, 16, 256, 256}},
    {1, 2, 0, {4, 6, 32, 32, 32}},     {1, 2, 0, {4, 6, 64, 64, 64}},
    {1, 2, 0, {4, 6, 16, 256, 256}},   {10, 2, 0, {4, 6, 16, 256, 256}},
    {100, 2, 0, {4, 6, 16, 256, 256}}, {1000, 2, 0, {4, 6, 16, 256, 256}},
};

#define INSTANTIATE_ERASE_SUITE(TEST, VALUES) \
  INSTANTIATE_TEST_SUITE_P(TEST, TEST ## Test, testing::ValuesIn(VALUES));

INSTANTIATE_ERASE_SUITE(EraseGpuKernel1f, values_1);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel2f, values_2);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel2NCf, values_2NC);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel3f, values_3);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel3fHWC, values_3HWC);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel3fCHW, values_3CHW);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel4fDHCW, values_4DHCW);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel4fDHWC, values_4DHWC);
INSTANTIATE_ERASE_SUITE(EraseGpuKernel5f, values_5);


}  // namespace kernels
}  // namespace dali
