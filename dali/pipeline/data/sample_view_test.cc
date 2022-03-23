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
#include <numeric>
#include <utility>

#include "dali/c_api.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_view.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {


template <typename SampleView>
void compare(const SampleView &sv, const void *ptr, const TensorShape<> &shape,
             DALIDataType dtype) {
  EXPECT_EQ(sv.raw_data(), ptr);
  EXPECT_EQ(sv.shape(), shape);
  EXPECT_EQ(sv.type(), dtype);
}


template <typename TensorView>
void compare(const TensorView &left, const TensorView &right) {
  EXPECT_EQ(left.data, right.data);
  EXPECT_EQ(left.shape, right.shape);
}


TEST(SampleView, Constructors) {
  SampleView<CPUBackend> default_view{};
  compare(default_view, nullptr, {0}, DALI_NO_TYPE);

  int32_t data{};
  SampleView<CPUBackend> from_ptr{&data, {1, 2, 3}};
  compare(from_ptr, &data, {1, 2, 3}, DALI_INT32);

  SampleView<CPUBackend> from_void_ptr{reinterpret_cast<void *>(42), {1, 2, 3}, DALI_FLOAT};
  compare(from_void_ptr, reinterpret_cast<void *>(42), {1, 2, 3}, DALI_FLOAT);
}


TEST(SampleView, ViewConversion) {
  int32_t data{};
  SampleView<CPUBackend> sample_view{&data, {1, 2, 3}};
  const SampleView<CPUBackend> const_sample_view{&data, {1, 2, 3}};

  compare(view<int32_t>(sample_view), TensorView<StorageCPU, int32_t>{&data, {1, 2, 3}});
  compare(view<int32_t, 3>(sample_view), TensorView<StorageCPU, int32_t, 3>{&data, {1, 2, 3}});
  compare(view<const int32_t>(sample_view),
          TensorView<StorageCPU, const int32_t>{&data, {1, 2, 3}});
  compare(view<const int32_t, 3>(sample_view),
          TensorView<StorageCPU, const int32_t, 3>{&data, {1, 2, 3}});

  compare(view<const int32_t>(const_sample_view),
          TensorView<StorageCPU, const int32_t>{&data, {1, 2, 3}});
  compare(view<const int32_t, 3>(const_sample_view),
          TensorView<StorageCPU, const int32_t, 3>{&data, {1, 2, 3}});
}

TEST(SampleView, ViewConversionError) {
  int32_t data{};
  SampleView<CPUBackend> sample_view{&data, {1, 2, 3}};
  const SampleView<CPUBackend> const_sample_view{&data, {1, 2, 3}};

  EXPECT_THROW(view<float>(sample_view), std::runtime_error);
  EXPECT_THROW(view<const float>(sample_view), std::runtime_error);
  EXPECT_THROW(view<const float>(const_sample_view), std::runtime_error);
}
}  // namespace dali
