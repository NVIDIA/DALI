// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <tuple>

#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/sequence_shape.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {

namespace sequence_utils_test {

template <typename BatchType>
struct BatchBackend;

template <template <typename> class BatchContainer, typename Backend>
struct BatchBackend<BatchContainer<Backend>> {
  using Type = Backend;
};

template <typename BatchType>
using batch_backend_t =
    typename BatchBackend<std::remove_cv_t<std::remove_reference_t<BatchType>>>::Type;

constexpr cudaStream_t cuda_stream = 0;

using namespace sequence_utils;  // NOLINT

TEST(SequenceShapeTest, FoldExtents1Unfolded) {
  TensorListShape<> frame_shape = {{4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5}, {101, 72}, {101, 72}};
  TensorListShape<> unfolded_extents = {{5}, {2}};
  TensorListShape<> expected_shape = {{5, 4, 5}, {2, 101, 72}};
  auto folded_shape = fold_outermost_like(frame_shape, unfolded_extents);
  EXPECT_EQ(folded_shape, expected_shape);
}

TEST(SequenceShapeTest, FoldExtents2Unfolded) {
  TensorListShape<> frame_shape = {{7, 1},    {7, 1},    {7, 1},  {7, 1},  {7, 1},  {7, 1},
                                   {101, 72}, {101, 72}, {8, 49}, {8, 49}, {8, 49}, {8, 49},
                                   {8, 49},   {8, 49},   {8, 49}, {8, 49}, {8, 49}, {8, 49},
                                   {8, 49},   {8, 49},   {5, 7},  {5, 7}};
  TensorListShape<> unfolded_extents = {{3, 2}, {2, 1}, {3, 4}, {1, 2}};
  TensorListShape<> expected_shape = {{3, 2, 7, 1}, {2, 1, 101, 72}, {3, 4, 8, 49}, {1, 2, 5, 7}};
  auto folded_shape = fold_outermost_like(frame_shape, unfolded_extents);
  EXPECT_EQ(folded_shape, expected_shape);
}

TEST(SequenceShapeTest, FoldExtents0Unfolded) {
  TensorListShape<> frame_shape = {{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {101, 72}, {101, 73}};
  TensorListShape<> unfolded_extents = {{}, {}, {}, {}, {}, {}, {}};
  auto folded_shape = fold_outermost_like(frame_shape, unfolded_extents);
  EXPECT_EQ(folded_shape, frame_shape);
}

TEST(SequenceShapeTest, FoldExtentsSingleFrame) {
  TensorListShape<> frame_shape = {{4224}};
  TensorListShape<> unfolded_extents = {{1, 1, 1, 1, 1}};
  TensorListShape<> expected_shape = {{1, 1, 1, 1, 1, 4224}};
  auto folded_shape = fold_outermost_like(frame_shape, unfolded_extents);
  EXPECT_EQ(folded_shape, expected_shape);
}

TEST(SequenceShapeTest, FoldExtentsZeroExtent) {
  TensorListShape<> frame_shape = {{1}, {1}, {1}, {1}, {1}};
  TensorListShape<> unfolded_extents = {{3, 1}, {5, 0}, {2, 1}};
  EXPECT_THROW(fold_outermost_like(frame_shape, unfolded_extents), std::runtime_error);
}

TEST(SequenceShapeTest, FoldExtentsNonUniformSample) {
  TensorListShape<> frame_shape = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {5, 6, 3},
                                   {5, 6, 3}, {5, 6, 3}, {5, 7, 3}};
  TensorListShape<> unfolded_extents = {{3, 1}, {2, 2}};
  EXPECT_THROW(fold_outermost_like(frame_shape, unfolded_extents), std::runtime_error);
}

void validate_range(UnfoldedSliceRange range, const uint8_t *base_ptr, TensorShape<> slice_shape,
                    size_t type_size, int num_slices) {
  auto stride = volume(slice_shape) * type_size;
  EXPECT_EQ(range.SliceShape(), slice_shape);
  EXPECT_EQ(range.SliceSize(), stride);
  ASSERT_EQ(range.NumSlices(), num_slices);
  int i = 0;
  for (auto &&slice : range) {
    EXPECT_EQ(slice.ptr, base_ptr + i++ * stride);
    EXPECT_EQ(slice.shape, slice_shape);
    EXPECT_EQ(slice.type_size, type_size);
  }
  EXPECT_EQ(i, num_slices);
}

TEST(SequenceShapeTest, Unfold0Extents) {
  uint8_t data{};
  TensorShape<> shape{3, 5, 7};
  size_t type_size = 4;
  SliceView view{&data, shape, type_size};
  validate_range(UnfoldedSliceRange(view, 0), &data, shape, type_size, 1);
}

TEST(SequenceShapeTest, Unfold1Extent) {
  uint8_t data{};
  TensorShape<> shape{3, 5, 7};
  size_t type_size = 8;
  SliceView view{&data, shape, type_size};
  validate_range(UnfoldedSliceRange(view, 1), &data, {5, 7}, type_size, 3);
}

TEST(SequenceShapeTest, Unfold2Extents) {
  uint8_t data{};
  TensorShape<> shape{3, 5, 7};
  size_t type_size = 1;
  SliceView view{&data, shape, type_size};
  validate_range(UnfoldedSliceRange(view, 2), &data, {7}, type_size, 15);
}

TEST(SequenceShapeTest, Unfold3Extents) {
  uint8_t data{};
  TensorShape<> shape{3, 5, 7};
  size_t type_size = 2;
  SliceView view{&data, shape, type_size};
  validate_range(UnfoldedSliceRange(view, 3), &data, {}, type_size, 105);
}

template <typename ContainerT>
void check_batch_props(const ContainerT &batch, const ContainerT &unfolded_batch, int num_slices,
                       int ndims_to_unfold) {
  ASSERT_EQ(batch.is_pinned(), unfolded_batch.is_pinned());
  ASSERT_EQ(batch.order(), unfolded_batch.order());
  ASSERT_EQ(batch.type(), unfolded_batch.type());
  ASSERT_EQ(unfolded_batch.num_samples(), num_slices);
  if (batch.GetLayout().empty()) {
    ASSERT_EQ(batch.GetLayout(), unfolded_batch.GetLayout());
  } else {
    ASSERT_EQ(batch.GetLayout().sub(ndims_to_unfold), unfolded_batch.GetLayout());
  }
}

template <typename Container>
class SequenceShapeUnfoldTest : public ::testing::Test {
 protected:
  std::tuple<Container, std::shared_ptr<Container>> CreateTestBatch(
      DALIDataType dtype, bool is_pinned = false, TensorLayout layout = "ABC",
      TensorListShape<> shape = {{3, 5, 7}, {11, 5, 4}, {7, 2, 11}}) {
    Container batch;
    constexpr bool is_device = std::is_same_v<batch_backend_t<Container>, GPUBackend>;
    batch.set_order(is_device ? AccessOrder(cuda_stream) : AccessOrder::host());
    batch.set_pinned(is_pinned);
    batch.Resize(shape, dtype);
    if (!layout.empty()) {
      batch.SetLayout(layout);
    }
    return {std::move(batch), expanded_like(batch)};
  }

  void TestUnfolding(const Container &batch, Container &unfolded_batch, int ndims_to_unfold) {
    const auto &shape = batch.shape();
    auto unfolded_extents = shape.first(ndims_to_unfold);
    auto slice_shapes = shape.last(shape.sample_dim() - ndims_to_unfold);
    auto unfolded_batch_size = unfolded_extents.num_elements();
    UnfoldExtents(batch, unfolded_batch, unfolded_extents);
    check_batch_props(batch, unfolded_batch, unfolded_batch_size, ndims_to_unfold);
    int slice_idx = 0;
    size_t type_size = batch.type_info().size();
    for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
      auto base_ptr = static_cast<const uint8_t *>(batch.raw_tensor(sample_idx));
      size_t stride = type_size * volume(slice_shapes[sample_idx]);
      for (int i = 0; i < volume(unfolded_extents[sample_idx]); i++) {
        auto ptr = static_cast<uint8_t *>(unfolded_batch.raw_mutable_tensor(slice_idx));
        EXPECT_EQ(ptr, base_ptr + stride * i);
        EXPECT_EQ(slice_shapes[sample_idx], unfolded_batch.tensor_shape(slice_idx++));
      }
    }
  }

  template <typename Backend>
  void UnfoldExtents(const TensorVector<Backend> &batch, TensorVector<Backend> &unfolded_batch,
                     TensorListShape<> unfolded_extents) {
    unfold_outer_dims(batch, unfolded_batch, unfolded_extents.sample_dim(),
                      unfolded_extents.num_elements());
  }

  template <typename Backend>
  void UnfoldExtents(const TensorList<Backend> &batch, TensorList<Backend> &unfolded_batch,
                     TensorListShape<> unfolded_extents) {
    unfold_outer_dims(batch, unfolded_batch, unfolded_extents.sample_dim());
  }
};

using Containers = ::testing::Types<TensorVector<CPUBackend>, TensorVector<GPUBackend>,
                                    TensorList<CPUBackend>, TensorList<GPUBackend>>;

TYPED_TEST_SUITE(SequenceShapeUnfoldTest, Containers);

TYPED_TEST(SequenceShapeUnfoldTest, Unfold0Extents) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT32);
  this->TestUnfolding(batch, *expanded_batch, 0);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold1Extent) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT16);
  this->TestUnfolding(batch, *expanded_batch, 1);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold2Extents) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8);
  this->TestUnfolding(batch, *expanded_batch, 2);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold2ExtentsPinned) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, true);
  this->TestUnfolding(batch, *expanded_batch, 2);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold2Extents3Iters) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, false, "XYZ");
  this->TestUnfolding(batch, *expanded_batch, 2);
  batch.Resize({{2, 2, 6}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("XYZ");
  this->TestUnfolding(batch, *expanded_batch, 2);
  batch.Resize({{2, 2, 6}, {13, 4, 11}, {13, 4, 11}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("XYZ");
  this->TestUnfolding(batch, *expanded_batch, 2);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold2Extents3ItersNoLayout) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, false, "");
  this->TestUnfolding(batch, *expanded_batch, 2);
  batch.Resize({{2, 2, 6}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("");
  this->TestUnfolding(batch, *expanded_batch, 2);
  batch.Resize({{2, 2, 6}, {13, 4, 11}, {13, 4, 11}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("");
  this->TestUnfolding(batch, *expanded_batch, 2);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold2ExtentsEmptyLayout) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, false, "");
  this->TestUnfolding(batch, *expanded_batch, 2);
}

TYPED_TEST(SequenceShapeUnfoldTest, Unfold3Extents) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_FLOAT);
  this->TestUnfolding(batch, *expanded_batch, 3);
}

template <typename Backend>
class SequenceShapeTVBroadcastTest : public ::testing::Test {
 protected:
  std::tuple<TensorVector<Backend>, std::shared_ptr<TensorVector<Backend>>> CreateTestBatch(
      DALIDataType dtype, bool is_pinned = false, TensorLayout layout = "ABC",
      TensorListShape<> shape = {{3, 5, 7}, {11, 5, 4}, {7, 2, 11}}) {
    TensorVector<Backend> batch;
    constexpr bool is_device = std::is_same_v<Backend, GPUBackend>;
    batch.set_order(is_device ? AccessOrder(cuda_stream) : AccessOrder::host());
    batch.set_pinned(is_pinned);
    batch.Resize(shape, dtype);
    if (!layout.empty()) {
      batch.SetLayout(layout);
    }
    return {std::move(batch), expanded_like(batch)};
  }

  void TestBroadcasting(const TensorVector<Backend> &batch, TensorVector<Backend> &expanded_batch,
                        const TensorListShape<> &expand_extents) {
    const auto &shape = batch.shape();
    auto expanded_batch_size = expand_extents.num_elements();
    broadcast_samples(batch, expanded_batch, expanded_batch_size, expand_extents);
    check_batch_props(batch, expanded_batch, expanded_batch_size, 0);
    for (int sample_idx = 0, elem_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
      const auto &sample_shape = shape[sample_idx];
      auto base_ptr = static_cast<const uint8_t *>(batch.raw_tensor(sample_idx));
      for (int i = 0; i < volume(expand_extents[sample_idx]); i++) {
        auto ptr = static_cast<uint8_t *>(expanded_batch.raw_mutable_tensor(elem_idx));
        EXPECT_EQ(ptr, base_ptr);
        EXPECT_EQ(sample_shape, expanded_batch.tensor_shape(elem_idx++));
      }
    }
  }
};

using Backends = ::testing::Types<CPUBackend, GPUBackend>;

TYPED_TEST_SUITE(SequenceShapeTVBroadcastTest, Backends);

TYPED_TEST(SequenceShapeTVBroadcastTest, Broadcast0Extents) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT32);
  this->TestBroadcasting(batch, *expanded_batch, {{}, {}, {}});
}

TYPED_TEST(SequenceShapeTVBroadcastTest, Broadcast1Extent) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT16);
  this->TestBroadcasting(batch, *expanded_batch, {{1}, {1}, {1}});
}

TYPED_TEST(SequenceShapeTVBroadcastTest, Broadcast2Extents) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8);
  this->TestBroadcasting(batch, *expanded_batch, {{5}, {3}, {1}});
}

TYPED_TEST(SequenceShapeTVBroadcastTest, Broadcast2ExtentsPinned) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, true);
  this->TestBroadcasting(batch, *expanded_batch, {{1, 1}, {7, 0}, {12, 4}});
}

TYPED_TEST(SequenceShapeTVBroadcastTest, Broadcast2Extents3Iters) {
  auto [batch, expanded_batch] = this->CreateTestBatch(DALI_UINT8, false, "XYZ");
  this->TestBroadcasting(batch, *expanded_batch, {{1, 1}, {7, 0}, {12, 4}});
  batch.Resize({{2, 2, 6}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("XYZ");
  this->TestBroadcasting(batch, *expanded_batch, {{5}, {1}});
  batch.Resize({{2, 2, 6}, {13, 4, 11}, {13, 4, 11}, {13, 4, 11}}, DALI_FLOAT);
  batch.SetLayout("XYZ");
  this->TestBroadcasting(batch, *expanded_batch, {{}, {}, {}, {}});
}

}  // namespace sequence_utils_test
}  // namespace dali
