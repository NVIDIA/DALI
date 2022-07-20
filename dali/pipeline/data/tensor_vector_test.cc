// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "dali/core/access_order.h"
#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/tensor_vector.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace test {



template <typename T>
class TensorVectorSuite : public ::testing::Test {
 protected:
  void validate(const TensorVector<T> &tv) {
    ASSERT_EQ(tv.size(), 2);
    EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
    EXPECT_EQ(tv[0]->shape(), TensorShape<>(2, 4));
    EXPECT_EQ(tv[1]->shape(), TensorShape<>(4, 2));
    EXPECT_EQ(tv[0]->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(tv[1]->nbytes(), 4 * 2 * sizeof(int32_t));
    EXPECT_EQ(tv.is_pinned(), false);
    EXPECT_EQ(tv[0]->is_pinned(), false);
    EXPECT_EQ(tv[1]->is_pinned(), false);
  }
};

typedef ::testing::Types<CPUBackend, GPUBackend> Backends;

constexpr cudaStream_t cuda_stream = 0;

TYPED_TEST_SUITE(TensorVectorSuite, Backends);

TYPED_TEST(TensorVectorSuite, NewSetupAndSetSizeNoncontiguous) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.set_order(order);
  tv.set_sample_dim(2);
  tv.SetLayout("XY");
  tv.SetContiguous(BatchState::Noncontiguous);
  tv.SetSize(3);


  auto empty_2d = TensorShape<>{0, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_NO_TYPE);
  }

  // Setting just the type
  tv.set_type(DALI_INT32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }

  Tensor<TypeParam> t;
  t.set_pinned(false);
  t.Resize({2, 3}, DALI_INT32);
  t.SetLayout("XY");

  // We need to propagate device id and order
  tv.set_device_id(t.device_id());
  tv.set_order(t.order());

  for (int i = 0; i < 3; i++) {
    tv.UnsafeSetSample(i, t);
    EXPECT_EQ(tv[i].raw_data(), t.raw_data());
    EXPECT_EQ(tv[i].shape(), t.shape());
    EXPECT_EQ(tv[i].type(), t.type());
  }

  tv.SetSize(4);
  // New one should be empty
  EXPECT_EQ(tv[3].raw_data(), nullptr);
  EXPECT_EQ(tv[3].shape(), empty_2d);
  EXPECT_EQ(tv[3].type(), DALI_INT32);

  tv.SetSize(2);
  tv.SetSize(3);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(tv[i].raw_data(), t.raw_data());
    EXPECT_EQ(tv[i].shape(), t.shape());
    EXPECT_EQ(tv[i].type(), t.type());
  }

  // As we were sharing, the share is removed and no allocation should happen
  for (int i = 2; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }

  // There should be no access to the out of bounds one
  EXPECT_THROW(tv[3], std::runtime_error);

  // We are sharing, no way to make it bigger
  EXPECT_THROW(tv.Resize(uniform_list_shape(3, {10, 12, 3})), std::runtime_error);
}


TYPED_TEST(TensorVectorSuite, NewSetupLikeMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  TensorVector<TypeParam> tv_like_t;
  tv_like_t.SetupLike(t);

  EXPECT_EQ(t.device_id(), tv_like_t.device_id());
  EXPECT_EQ(t.order(), tv_like_t.order());
  EXPECT_EQ(t.is_pinned(), tv_like_t.is_pinned());
  EXPECT_EQ(t.shape().sample_dim(), tv_like_t.sample_dim());
  EXPECT_EQ(t.GetLayout(), tv_like_t.GetLayout());

  TensorVector<TypeParam> tv_like_tv;
  tv_like_tv.SetupLike(tv_like_t);

  EXPECT_EQ(tv_like_t.device_id(), tv_like_tv.device_id());
  EXPECT_EQ(tv_like_t.order(), tv_like_tv.order());
  EXPECT_EQ(tv_like_t.is_pinned(), tv_like_tv.is_pinned());
  EXPECT_EQ(tv_like_t.sample_dim(), tv_like_tv.sample_dim());
  EXPECT_EQ(tv_like_t.GetLayout(), tv_like_tv.GetLayout());
}

template <typename Backend>
std::vector<std::pair<std::string, std::function<void(TensorVector<Backend> &)>>>
SetRequiredSetters(int sample_dim, DALIDataType type, TensorLayout layout, bool pinned,
                   int device_id) {
  return {
    {"sample dim", [sample_dim](TensorVector<Backend> &t) {
      t.set_sample_dim(sample_dim);
    }},
    {"type", [type](TensorVector<Backend> &t) {
      t.set_type(type);
    }},
    {"layout", [layout](TensorVector<Backend> &t) {
      t.SetLayout(layout);
    }},
    {"device id", [device_id](TensorVector<Backend> &t) {
      t.set_device_id(device_id);
    }},
    {"pinned", [pinned](TensorVector<Backend> &t) {
      t.set_pinned(pinned);
    }},
    {"order", [device_id](TensorVector<Backend> &t) {
      constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
      const auto order = is_device ? AccessOrder(cuda_stream, device_id) : AccessOrder::host();
      t.set_order(order);
    }},
  };
}

TYPED_TEST(TensorVectorSuite, NewPartialSetupSetMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = SetRequiredSetters<TypeParam>(3, DALI_INT32, "HWC", true, device);
  for (size_t excluded = 0; excluded < setups.size(); excluded++) {
    std::vector<size_t> idxs(setups.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    do {
      TensorVector<TypeParam> tv;
      tv.SetContiguous(BatchState::Noncontiguous);
      tv.SetSize(4);
      tv.set_pinned(false);
      for (auto idx : idxs) {
        if (idx == excluded) {
          continue;
        }
        setups[idx].second(tv);
      }
      try {
        tv.UnsafeSetSample(0, t);
        FAIL() << "Exception was expected with excluded: " << setups[excluded].first;
      } catch(std::runtime_error &e) {
        auto expected = "Sample must have the same " + setups[excluded].first;
        EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
            << expected << "\n====\nvs\n====\n"
            << e.what();
      } catch (...) {
        FAIL() << "Unexpected exception";
      }
    } while (std::next_permutation(idxs.begin(), idxs.end()));
  }
}


template <typename Backend>
std::vector<std::pair<std::string, std::function<void(TensorVector<Backend> &)>>>
CopyRequiredSetters(int sample_dim, DALIDataType type, TensorLayout layout) {
  return {
    {"sample dim", [sample_dim](TensorVector<Backend> &t) {
      t.set_sample_dim(sample_dim);
    }},
    {"type", [type](TensorVector<Backend> &t) {
      t.set_type(type);
    }},
    {"layout", [layout](TensorVector<Backend> &t) {
      t.SetLayout(layout);
    }}
  };
}


TYPED_TEST(TensorVectorSuite, NewPartialSetupCopyMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = CopyRequiredSetters<TypeParam>(3, DALI_INT32, "HWC");
  for (size_t excluded = 0; excluded < setups.size(); excluded++) {
    std::vector<size_t> idxs(setups.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    do {
      TensorVector<TypeParam> tv;
      tv.SetContiguous(BatchState::Noncontiguous);
      tv.SetSize(4);
      for (auto idx : idxs) {
        if (idx == excluded) {
          continue;
        }
        setups[idx].second(tv);
      }
      try {
        tv.UnsafeCopySample(0, t);
        FAIL() << "Exception was expected with excluded: " << setups[excluded].first;
      } catch(std::runtime_error &e) {
        auto expected = "Sample must have the same " + setups[excluded].first;
        EXPECT_NE(std::string(e.what()).rfind(expected), std::string::npos)
            << expected << "\n====\nvs\n====\n"
            << e.what();
      } catch (...) {
        FAIL() << "Unexpected exception";
      }
    } while (std::next_permutation(idxs.begin(), idxs.end()));
  }
}


TYPED_TEST(TensorVectorSuite, NewFullSetupSetMultiGPU) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr int device = 1;
  const auto order = is_device ? AccessOrder(cuda_stream, device) : AccessOrder::host();
  Tensor<TypeParam> t;
  t.set_device_id(device);
  t.set_order(order);
  t.set_pinned(true);
  t.Resize({3, 4, 5}, DALI_INT32);
  t.SetLayout("HWC");

  // set size to be checked. Copy doesn't make sense
  auto setups = SetRequiredSetters<TypeParam>(3, DALI_INT32, "HWC", true, device);
  std::vector<size_t> idxs(setups.size());
  std::iota(idxs.begin(), idxs.end(), 0);
  do {
    TensorVector<TypeParam> tv;
    tv.SetContiguous(BatchState::Noncontiguous);
    tv.SetSize(4);
    for (auto idx : idxs) {
      setups[idx].second(tv);
    }
    tv.UnsafeSetSample(0, t);
    EXPECT_EQ(tv[0].raw_data(), t.raw_data());
  } while (std::next_permutation(idxs.begin(), idxs.end()));
}


TYPED_TEST(TensorVectorSuite, NewResizeSetSize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.set_order(order);
  tv.set_sample_dim(2);
  tv.SetLayout("XY");
  tv.SetContiguous(BatchState::Contiguous);
  tv.SetSize(3);


  auto empty_2d = TensorShape<>{0, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_NO_TYPE);
  }

  // Setting just the type
  tv.set_type(DALI_INT32);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_2d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape);
  tv.SetLayout("HWC");

  const auto *base = static_cast<const uint8_t*>(unsafe_sample_owner(tv, 0).get());

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
    base += sizeof(int32_t) * new_shape[i].num_elements();
  }

  tv.SetSize(4);

  auto empty_3d = TensorShape<>{0, 0, 0};

  // New one should be empty
  EXPECT_EQ(tv[3].raw_data(), nullptr);
  EXPECT_EQ(tv[3].shape(), empty_3d);
  EXPECT_EQ(tv[3].type(), DALI_INT32);

  tv.SetSize(2);
  tv.SetSize(3);

  base = static_cast<const uint8_t*>(unsafe_sample_owner(tv, 0).get());

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
    base += sizeof(int32_t) * new_shape[i].num_elements();
  }

  // As we contiguous, thus sharing the contiguous buffer via sample, the share is removed
  // and no allocation should happen
  for (int i = 2; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), empty_3d);
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}


TYPED_TEST(TensorVectorSuite, NewNoForcedChangeContToNon) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Contiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  EXPECT_THROW(tv.Resize(new_shape, DALI_FLOAT, BatchState::Noncontiguous), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, NewNoForcedChangeNonToCont) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Noncontiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  EXPECT_THROW(tv.Resize(new_shape, DALI_FLOAT, BatchState::Contiguous), std::runtime_error);
}


TYPED_TEST(TensorVectorSuite, NewContiguousResize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<TypeParam> tv;
  tv.set_order(order);
  tv.SetContiguous(BatchState::Contiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    tv.UnsafeCopySample(i, tv, i);
  }

  const auto *base = static_cast<const uint8_t*>(unsafe_sample_owner(tv, 0).get());

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    base += sizeof(float) * new_shape[i].num_elements();
  }

  // Cannot copy without exact shape match when contiguous
  EXPECT_THROW(tv.UnsafeCopySample(0, tv, 1), std::runtime_error);
  EXPECT_THROW(tv.UnsafeCopySample(2, tv, 1), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, NewNoncontiguousResize) {
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<TypeParam> tv;
  tv.set_order(order);
  tv.SetContiguous(BatchState::Noncontiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Cannot copy without exact shape match when contiguous
  tv.UnsafeCopySample(0, tv, 1);
  tv.UnsafeCopySample(2, tv, 1);

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(tv[i].shape(), new_shape[1]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }
}


TEST(TensorVectorSuite, NewContiguousResizeCpu) {
  using TypeParam = CPUBackend;
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Contiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    ConstantFill(view<float, 3>(tv[i]), i);
  }

  for (int i = 0; i < 3; i++) {
    tv.UnsafeCopySample(i, tv, i);
  }

  const auto *base = static_cast<const uint8_t*>(unsafe_sample_owner(tv, 0).get());

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(tv[i].raw_data(), base);
    EXPECT_EQ(*tv.template tensor<float>(i), i);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
    base += sizeof(float) * new_shape[i].num_elements();
  }

  // Cannot copy without exact shape match when contiguous
  EXPECT_THROW(tv.UnsafeCopySample(0, tv, 1), std::runtime_error);
  EXPECT_THROW(tv.UnsafeCopySample(2, tv, 1), std::runtime_error);
}

TEST(TensorVectorSuite, NewNoncontiguousResizeCpu) {
  using TypeParam = CPUBackend;
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Noncontiguous);
  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT);

  for (int i = 0; i < 3; i++) {
    ConstantFill(view<float, 3>(tv[i]), i);
  }

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(*tv.template tensor<float>(i), i);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  // Cannot copy without exact shape match when contiguous
  tv.UnsafeCopySample(0, tv, 1);
  tv.UnsafeCopySample(2, tv, 1);

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(*tv.template tensor<float>(i), 1);
    EXPECT_EQ(tv[i].shape(), new_shape[1]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  tv.SetSize(5);
  tv.UnsafeSetSample(3, tv, 0);
  tv.UnsafeSetSample(4, tv, 0);
  for (int i = 3; i < 5; i++) {
    EXPECT_EQ(tv[i].raw_data(), tv[0].raw_data());
    EXPECT_EQ(*tv.template tensor<float>(i), 1);
    EXPECT_EQ(tv[i].shape(), new_shape[1]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }
}

TEST(TensorVectorSuite, NewBreakContiguity) {
  using TypeParam = CPUBackend;
  TensorVector<TypeParam> tv;
  // anything goes
  tv.SetContiguous(BatchState::Default);

  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  tv.Resize(new_shape, DALI_FLOAT, BatchState::Contiguous);

  for (int i = 0; i < 3; i++) {
    ConstantFill(view<float, 3>(tv[i]), i);
  }

  for (int i = 0; i < 3; i++) {
    EXPECT_NE(tv[i].raw_data(), nullptr);
    EXPECT_EQ(*tv.template tensor<float>(i), i);
    EXPECT_EQ(tv[i].shape(), new_shape[i]);
    EXPECT_EQ(tv[i].type(), DALI_FLOAT);
  }

  Tensor<TypeParam> t;
  t.Resize({2, 3, 4}, DALI_FLOAT);
  ConstantFill(view<float, 3>(t), 42);

  tv.UnsafeSetSample(1, t);
  EXPECT_FALSE(tv.IsContiguous());

  EXPECT_NE(tv[1].raw_data(), nullptr);
  EXPECT_EQ(*tv.template tensor<float>(1), 42);
  EXPECT_EQ(tv[1].shape(), t.shape());
  EXPECT_EQ(tv[1].type(), DALI_FLOAT);
}


TYPED_TEST(TensorVectorSuite, NewReserve) {
  // Verify that we still keep the memory reserved in sample mode
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Default);
  tv.reserve(100, 4);

  auto new_shape = TensorListShape<>{{1, 2, 3}, {2, 3, 4}, {3, 4, 50}};
  tv.Resize(new_shape, DALI_FLOAT, BatchState::Noncontiguous);

  tv.SetSize(4);
  auto capacity = tv._chunks_capacity();

  for (int i = 0; i < 4; i++) {
    auto expected_capacity =
        std::max<int>(100, i < 3 ? sizeof(float) * new_shape[i].num_elements() : 0);
    EXPECT_EQ(capacity[i], expected_capacity);
  }
}

// Check if interleaving any of
// * set_pinned
// * set_type
// * Resize(shape)
// * Resize(shape, type)
// * reserve
// behaves as it is supposed to - that is set_type always first, set_type before Reshape,
// reserve can be without type.

// TODO(klecki): reverse pinned and capacity tests

TYPED_TEST(TensorVectorSuite, PinnedAfterReserveThrows) {
  TensorVector<TypeParam> tv_0, tv_1;
  tv_0.reserve(100);
  EXPECT_THROW(tv_0.set_pinned(false), std::runtime_error);
  tv_1.reserve(100, 2);
  EXPECT_THROW(tv_1.set_pinned(false), std::runtime_error);
  TensorVector<TypeParam> tv_2(2), tv_3(2);
  tv_2.reserve(100);
  EXPECT_THROW(tv_2.set_pinned(false), std::runtime_error);
  tv_3.reserve(100, 2);
  EXPECT_THROW(tv_3.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedAfterResizeThrows) {
  TensorVector<TypeParam> tv;
  tv.reserve(100);
  EXPECT_THROW(tv.Resize({{2, 4}, {4, 2}}), std::runtime_error);
  tv.Resize({{2, 4}, {4, 2}}, DALI_INT32);
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv[0].type(), DALI_INT32);
  EXPECT_EQ(tv[1].type(), DALI_INT32);
  ASSERT_THROW(tv.set_pinned(false), std::runtime_error);
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeContiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(100);
  EXPECT_THROW(tv.Resize({{2, 4}, {4, 2}}), std::runtime_error);
  tv.template set_type<int32_t>();
  tv.Resize({{2, 4}, {4, 2}});
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv.is_pinned(), false);
  for (int i = 0; i < tv.num_samples(); i++) {
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}

TYPED_TEST(TensorVectorSuite, PinnedBeforeResizeNoncontiguous) {
  TensorVector<TypeParam> tv;
  tv.set_pinned(false);
  tv.reserve(50, 2);
  tv.template set_type<int32_t>();
  tv.Resize({{2, 4}, {4, 2}});
  ASSERT_EQ(tv.num_samples(), 2);
  EXPECT_EQ(tv.shape(), TensorListShape<>({{2, 4}, {4, 2}}));
  EXPECT_EQ(tv[0].shape(), TensorShape<>(2, 4));
  EXPECT_EQ(tv[1].shape(), TensorShape<>(4, 2));
  EXPECT_EQ(tv.is_pinned(), false);
  for (int i = 0; i < tv.num_samples(); i++) {
    EXPECT_EQ(tv[i].type(), DALI_INT32);
  }
}

namespace {

const uint8_t kMagicNumber = 42;

void FillWithMagicNumber(TensorVector<CPUBackend> &tv) {
  for (int i = 0; i < tv.shape().num_elements(); i++) {
    // We utilize the contiguity of the TensorVector
    tv.mutable_tensor<uint8_t>(0)[i] = kMagicNumber;
  }
}

void FillWithMagicNumber(TensorVector<GPUBackend> &tv) {
  // We utilize the contiguity of the TensorVector
  cudaMemset(tv.mutable_tensor<uint8_t>(0), kMagicNumber, tv.shape().num_elements());
}


void CompareWithMagicNumber(Tensor<CPUBackend> &t) {
  for (int i = 0; i < t.shape().num_elements(); i++) {
    EXPECT_EQ(t.data<uint8_t>()[i], kMagicNumber);
  }
}

void CompareWithMagicNumber(Tensor<GPUBackend> &t) {
  uint8_t *ptr;
  std::vector<uint8_t> buffer(t.shape().num_elements());
  cudaMemcpy(buffer.data(), t.data<uint8_t>(), t.shape().num_elements(), cudaMemcpyDeviceToHost);
  for (auto b : buffer) {
    EXPECT_EQ(b, kMagicNumber);
  }
}


template <typename Backend>
Tensor<Backend> ReturnTvAsTensor() {
  TensorVector<Backend> tv;
  tv.SetContiguous(BatchState::Contiguous);
  tv.Resize(uniform_list_shape(4, {1, 2, 3}), DALI_UINT8);
  tv.SetLayout("HWC");
  FillWithMagicNumber(tv);
  return tv.AsTensor();
}

}  // namespace


TYPED_TEST(TensorVectorSuite, TensorVectorAsTensorAccess) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Contiguous);
  auto shape = TensorListShape<>{{1, 2, 3}, {1, 2, 4}};
  tv.Resize(shape, DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousTensor());
  EXPECT_FALSE(tv.IsDenseTensor());

  {
    auto tensor_shape = TensorShape<>{2, 7};
    auto tensor = tv.AsReshapedTensor(tensor_shape);
    EXPECT_EQ(tensor.shape(), tensor_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }
  tv.Resize(uniform_list_shape(3, {2, 3, 4}));
  tv.SetLayout("HWC");

  EXPECT_TRUE(tv.IsContiguousTensor());
  EXPECT_TRUE(tv.IsDenseTensor());

  {
    auto expected_shape = TensorShape<>{3, 2, 3, 4};
    auto tensor = tv.AsTensor();
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.GetLayout(), "NHWC");
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }

  {
    // Verify that we can convert and touch the data after TV is already released
    auto tensor = ReturnTvAsTensor<TypeParam>();
    auto expected_shape = TensorShape<>{4, 1, 2, 3};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_UINT8);
    EXPECT_EQ(tensor.GetLayout(), "NHWC");
    CompareWithMagicNumber(tensor);
  }
}

TYPED_TEST(TensorVectorSuite, EmptyTensorVectorAsTensorAccess) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Contiguous);
  auto shape = TensorListShape<>{{1, 2, 0, 3}, {1, 2, 0, 3}};
  tv.Resize(shape, DALI_INT32);
  EXPECT_TRUE(tv.IsContiguousTensor());
  EXPECT_TRUE(tv.IsDenseTensor());

  {
    auto tensor_shape = TensorShape<>{1, 0};
    auto tensor = tv.AsReshapedTensor(tensor_shape);
    EXPECT_EQ(tensor.shape(), tensor_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.raw_data(), nullptr);
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }

  {
    auto expected_shape = TensorShape<>{2, 1, 2, 0, 3};
    auto tensor = tv.AsTensor();
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.type(), DALI_INT32);
    EXPECT_EQ(tensor.raw_data(), nullptr);
    EXPECT_EQ(tensor.raw_data(), tv.raw_tensor(0));
  }
}


TYPED_TEST(TensorVectorSuite, BatchResize) {
  TensorVector<TypeParam> tv(5);
  ASSERT_EQ(tv.num_samples(), 5);
  tv.reserve(100);
  tv.reserve(200);
  tv.template set_type<int32_t>();
  tv.Resize(uniform_list_shape(5, {10, 20}));
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeDown) {
  TensorVector<TypeParam> tv(32);
  ASSERT_EQ(tv.num_samples(), 32);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorVectorSuite, VariableBatchResizeUp) {
  TensorVector<TypeParam> tv(2);
  ASSERT_EQ(tv.num_samples(), 2);
  TensorListShape<> new_size = {{42}, {42}, {42}, {42}, {42}};
  tv.Resize(new_size, DALI_UINT8);
  ASSERT_EQ(tv.num_samples(), new_size.num_samples());
}

TYPED_TEST(TensorVectorSuite, EmptyShareContiguous) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Contiguous);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorVector<TypeParam> target;
  target.ShareData(tv);
  ASSERT_EQ(target.num_samples(), shape.num_samples());
  ASSERT_EQ(target.type(), DALI_UINT8);
  ASSERT_EQ(target.shape(), shape);
  ASSERT_TRUE(target.IsContiguous());
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(target.raw_tensor(i), nullptr);
    ASSERT_EQ(target.raw_tensor(i), tv.raw_tensor(i));
  }
}

TYPED_TEST(TensorVectorSuite, EmptyShareNonContiguous) {
  TensorVector<TypeParam> tv;
  tv.SetContiguous(BatchState::Noncontiguous);
  TensorListShape<> shape = {{100, 0, 0}, {42, 0, 0}};
  tv.Resize(shape, DALI_UINT8);
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(tv.raw_tensor(i), nullptr);
  }

  TensorVector<TypeParam> target;
  target.ShareData(tv);
  ASSERT_EQ(target.num_samples(), shape.num_samples());
  ASSERT_EQ(target.type(), DALI_UINT8);
  ASSERT_EQ(target.shape(), shape);
  ASSERT_FALSE(target.IsContiguous());
  for (int i = 0; i < shape.num_samples(); i++) {
    ASSERT_EQ(target.raw_tensor(i), nullptr);
    ASSERT_EQ(target.raw_tensor(i), tv.raw_tensor(i));
  }
}

template <typename Backend, typename F>
void test_moving_props(const bool is_pinned, const TensorLayout layout,
                       const TensorListShape<> shape, const int sample_dim, const DALIDataType type,
                       F &&mover) {
  constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
  const auto order = is_device ? AccessOrder(cuda_stream) : AccessOrder::host();
  TensorVector<Backend> tv;
  tv.set_pinned(is_pinned);
  tv.set_order(order);
  tv.set_sample_dim(sample_dim);
  tv.Resize(shape, type);
  tv.SetLayout(layout);

  const auto check = [&](auto &moved) {
    EXPECT_EQ(moved.order(), order);
    EXPECT_EQ(moved.GetLayout(), layout);
    EXPECT_EQ(moved.sample_dim(), sample_dim);
    EXPECT_EQ(moved.shape(), shape);
    EXPECT_EQ(moved.type(), type);
    EXPECT_EQ(moved.is_pinned(), is_pinned);
  };
  mover(check, tv);
}

TYPED_TEST(TensorVectorSuite, MoveConstructorMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorVector<TypeParam> moved{std::move(tv)};
                                 check(moved);
                               });
}

TYPED_TEST(TensorVectorSuite, MoveAssignmentMetaData) {
  test_moving_props<TypeParam>(true, "XYZ",
                               {{42, 1, 2}, {1, 2, 42}, {2, 42, 1}, {1, 42, 1}, {2, 42, 2}}, 3,
                               DALI_UINT16, [](auto &&check, auto &tv) {
                                 TensorVector<TypeParam> moved(2);
                                 moved = std::move(tv);
                                 check(moved);
                               });
}

TYPED_TEST(TensorVectorSuite, DeviceIdPropagationMultiGPU) {
  int num_devices = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  }
  constexpr bool is_device = std::is_same_v<TypeParam, GPUBackend>;
  constexpr bool is_pinned = !is_device;
  AccessOrder order = AccessOrder::host();
  TensorShape<> shape{42};
  for (int device_id = 0; device_id < num_devices; device_id++) {
    TensorVector<TypeParam> batch;
    batch.SetSize(1);
    DeviceGuard dg(device_id);
    batch.set_device_id(device_id);
    batch.set_pinned(is_pinned);
    batch.set_type(DALI_UINT8);
    batch.set_sample_dim(shape.sample_dim());
    batch.set_order(order);
    void *data_ptr;
    std::shared_ptr<void> ptr;
    if (is_device) {
      CUDA_CALL(cudaMalloc(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFree(ptr); });
    } else {
      CUDA_CALL(cudaMallocHost(&data_ptr, shape.num_elements() * sizeof(uint8_t)));
      ptr = std::shared_ptr<void>(data_ptr, [](void *ptr) { cudaFreeHost(ptr); });
    }
    batch.UnsafeSetSample(0, ptr, shape.num_elements() * sizeof(uint8_t), is_pinned, shape,
                          DALI_UINT8, device_id, order);
    ASSERT_EQ(batch.device_id(), device_id);
    ASSERT_EQ(batch.order().device_id(), AccessOrder::host().device_id());
    ASSERT_NE(batch.order().device_id(), batch.device_id());
  }
}

namespace {

/**
 * GTest predicate formatter. Compares a batch of data contained in TensorList or TensorVector
 * @tparam T TensorList<CPUBackend> or TensorVector<CPUBackend>
 * @tparam U TensorList<CPUBackend> or TensorVector<CPUBackend>
 */
template <typename T, typename U>
::testing::AssertionResult Compare(const char *rhs_expr, const char *lhs_expr, const T &rhs,
                                   const U &lhs) {
  static_assert(std::is_same<T, TensorList<CPUBackend>>::value ||
                    std::is_same<T, TensorVector<CPUBackend>>::value,
                "T must be either TensorList<CPUBackend> or TensorVector<CPUBackend>");
  static_assert(std::is_same<U, TensorList<CPUBackend>>::value ||
                    std::is_same<U, TensorVector<CPUBackend>>::value,
                "U must be either TensorList<CPUBackend> or TensorVector<CPUBackend>");
  std::string testing_values = make_string(rhs_expr, ", ", lhs_expr);
  if (rhs.num_samples() != lhs.num_samples()) {
    ::testing::AssertionFailure() << make_string("[Testing: ", testing_values,
                                                 "] Inconsistent number of tensors");
  }
  for (int tensor_idx = 0; tensor_idx < rhs.num_samples(); tensor_idx++) {
    if (rhs.tensor_shape(tensor_idx) != lhs.tensor_shape(tensor_idx)) {
      ::testing::AssertionFailure()
          << make_string("[Testing: ", testing_values, "] Inconsistent shapes");
    }
    auto vol = volume(rhs.tensor_shape(tensor_idx));
    auto lptr = reinterpret_cast<const char *>(lhs.raw_tensor(tensor_idx));
    auto rptr = reinterpret_cast<const char *>(rhs.raw_tensor(tensor_idx));
    for (int i = 0; i < vol; i++) {
      if (rptr[i] != lptr[i]) {
        return ::testing::AssertionFailure()
               << make_string("[Testing: ", testing_values, "] Values at index [", i,
                              "] don't match: ", static_cast<int>(rptr[i]), " vs ",
                              static_cast<int>(lptr[i]), "");
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

class TensorVectorVariableBatchSizeTest : public ::testing::Test {
 protected:
  void SetUp() final {
    GenerateTestTv();
    GenerateTestTl();
  }

  void GenerateTestTv() {
    test_tv_.Resize(shape_, DALI_FLOAT);
    for (int i = 0; i < shape_.num_samples(); i++) {
      UniformRandomFill(view<float>(test_tv_[i]), rng_, 0.f, 1.f);
    }
  }

  void GenerateTestTl() {
    test_tl_.Resize(shape_, DALI_FLOAT);
    for (int i = 0; i < shape_.num_samples(); i++) {
      UniformRandomFill(view<float>(test_tl_), rng_, 0.f, 1.f);
    }
  }

  std::mt19937 rng_;
  TensorListShape<> shape_ = {{7, 2, 3}, {6, 1, 6}, {9, 3, 6}, {2, 9, 4},
                              {2, 3, 7}, {5, 3, 5}, {1, 1, 1}, {8, 3, 5}};
  TensorList<CPUBackend> test_tl_;
  TensorVector<CPUBackend> test_tv_;
};

TEST_F(TensorVectorVariableBatchSizeTest, SelfTest) {
  for (int i = 0; i < shape_.num_samples(); i++) {
    EXPECT_EQ(test_tv_.tensor_shape(i), shape_[i]);
    EXPECT_EQ(test_tl_.tensor_shape(i), shape_[i]);
  }
  EXPECT_PRED_FORMAT2(Compare, test_tv_, test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, test_tl_);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvShareWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlShareWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.ShareData(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TvCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tv_);
  EXPECT_PRED_FORMAT2(Compare, test_tv_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeUp) {
  TensorVector<CPUBackend> tv(2);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

TEST_F(TensorVectorVariableBatchSizeTest, TlCopyWithResizeDown) {
  TensorVector<CPUBackend> tv(32);
  tv.Copy(this->test_tl_);
  EXPECT_PRED_FORMAT2(Compare, test_tl_, tv);
}

}  // namespace test
}  // namespace dali
