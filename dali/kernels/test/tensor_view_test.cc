// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/tensor_view.h"

namespace dali {
namespace kernels {

TEST(ShapeDimTest, Value) {
  int x1[] = {1, 2, 3};
  EXPECT_EQ(ShapeDim(x1), 3);
  std::array<int, 5> x2;
  EXPECT_EQ(ShapeDim(x2), 5);
  std::vector<int> x3 = {1, 2, 3, 4};
  EXPECT_EQ(ShapeDim(x3), 4);
}

TEST(CompileTimeSize, DimInference) {
  std::array<int, 3> a;
  const auto *pa = &a;
  static_assert(compile_time_size<decltype(a)>::value == static_cast<int>(a.size()),
    "Compile-time size of an std-array should match its size argument");
  static_assert(compile_time_size<decltype(*pa)>::value == static_cast<int>(a.size()),
    "Compile-time size of an std-array should match its size argument");

  int x[4];
  static_assert(compile_time_size<decltype(x)>::value == 4,
    "Compile-time size of an array should should match its ndim");
  const int y[5] = { 9, 8, 7, 6, 5 };  // check that the  trait can strip const qualifier
  static_assert(compile_time_size<decltype(y)>::value == 5,
    "Compile-time size of an array should should match its ndim");

  volatile TensorShape<6> t;  // to check that the trait can strip volatile qualifier
  static_assert(compile_time_size<decltype(t)>::value == 6,
    "Compile-time of a static-ndim TensorShape should match its ndim");
}

TEST(TensorViewTest, StaticConstructors) {
  TensorView<EmptyBackendTag, int, 4> empty_static_dim{};
  EXPECT_EQ(empty_static_dim.data, nullptr);
  EXPECT_EQ(empty_static_dim.shape, TensorShape<4>());
}

TEST(TensorViewTest, Conversions) {
  TensorView<EmptyBackendTag, int, 4> static_dim{static_cast<int*>(nullptr), {1, 2, 3, 4}};
  ASSERT_EQ(static_dim.dim(), 4);
  // Allowed conversions
  TensorView<EmptyBackendTag, int, DynamicDimensions> dynamic_dim{static_dim};
  EXPECT_EQ(dynamic_dim.shape, static_dim.shape);
  ASSERT_EQ(dynamic_dim.dim(), 4);
  TensorView<EmptyBackendTag, int, 4> static_dim_2(dynamic_dim.to_static<4>());
  EXPECT_EQ(static_dim_2.shape, static_dim.shape);
  EXPECT_EQ(static_dim_2.shape, dynamic_dim.shape);

  dynamic_dim = TensorView<EmptyBackendTag, int, 2>{static_cast<int*>(nullptr), {1, 2}};
  ASSERT_EQ(dynamic_dim.dim(), 2);
}

TEST(TensorViewTest, Addressing) {
  TensorView<EmptyBackendTag, int, 3> tv{static_cast<int*>(nullptr), {4, 100, 50}};
  EXPECT_EQ(tv(0, 0, 0), static_cast<int*>(nullptr));
  EXPECT_EQ(tv(0, 0, 1), static_cast<int*>(nullptr) + 1);
  EXPECT_EQ(tv(0, 1, 0), static_cast<int*>(nullptr) + 50);
  EXPECT_EQ(tv(1, 0, 0), static_cast<int*>(nullptr) + 5000);
  EXPECT_EQ(tv(1, 1, 1), static_cast<int*>(nullptr) + 5051);
  EXPECT_EQ(tv(1, 1), static_cast<int*>(nullptr) + 5050);
  // EXPECT_EQ(tv(1), static_cast<int*>(nullptr) + 5000); // TODO - this is ambigous
}

TEST(TensorViewTest, TypePromotion) {
  TensorView<EmptyBackendTag, int, 3> tv{nullptr, {1, 2, 3}};
  TensorView<EmptyBackendTag, const int, 3> tvc = tv;
  EXPECT_EQ(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, tv.data);
  tvc = {};
  EXPECT_NE(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, nullptr);
  tvc = tv;
  EXPECT_EQ(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, tv.data);

  TensorView<EmptyBackendTag, int> tv_dyn = tv;
  EXPECT_EQ(tv_dyn.shape, tv.shape);
  EXPECT_EQ(tv_dyn.data, tv.data);

  TensorView<EmptyBackendTag, const int> tvc_dyn = tv;
  EXPECT_EQ(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, tv.data);
  tvc_dyn = {};
  EXPECT_NE(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, nullptr);
  tvc_dyn = tv;
  EXPECT_EQ(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, tv.data);

  auto *ptr = tv_dyn.shape.shape.data();
  tvc_dyn = std::move(tv_dyn);
  EXPECT_EQ(tvc_dyn.shape.shape.data(), ptr) << "Move is broken - a copy appeared somewhere.";
}

TEST(TensorListViewTest, Constructor) {
  TensorListView<EmptyBackendTag, int, 3> tlv{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  ASSERT_EQ(tlv.size(), 4);
  ASSERT_EQ(tlv.sample_dim(), 3);
  TensorListView<EmptyBackendTag, int> tlv_dynamic(tlv);
  ASSERT_EQ(tlv_dynamic.size(), 4);
  ASSERT_EQ(tlv_dynamic.sample_dim(), 3);
}

TEST(TensorListViewTest, OperatorSubscript) {
  TensorListView<EmptyBackendTag, int, 3> tlv{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  EXPECT_EQ(tlv[0].shape.size(), 3);
  EXPECT_EQ(tlv[0].data, static_cast<int*>(nullptr));
  EXPECT_EQ(tlv[1].data, static_cast<int*>(nullptr) + 4 * 100 * 50);
  EXPECT_EQ(tlv[2].data, static_cast<int*>(nullptr) + 4 * 100 * 50 + 2 * 10 * 5);
  EXPECT_EQ(tlv[3].data, static_cast<int*>(nullptr) + 4 * 100 * 50 + 2 * 10 * 5 + 4 * 50 * 25);
}

TEST(TensorListViewTest, ObtainingTensorViewFromStatic) {
  TensorListView<EmptyBackendTag, int, 3> tlv_static{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};

  auto t0 = tlv_static[0];
  static_assert(std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, 3>>::value,
                "Wrong type");
  auto t1 = tlv_static.tensor_view<3>(1);
  static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value,
                "Wrong type");
  auto t2 = tlv_static.tensor_view<DynamicDimensions>(2);
  static_assert(
      std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
      "Wrong type");
  EXPECT_EQ(t2.dim(), 3);
}

TEST(TensorListViewTest, ObtainingTensorViewFromDynamic) {
  TensorListView<EmptyBackendTag, int> tlv_dynamic{
      static_cast<int*>(nullptr), {{4, 100, 50}, {2, 10, 5}, {4, 50, 25}, {4, 100, 50}}};
  auto t0 = tlv_dynamic[0];
  static_assert(
      std::is_same<decltype(t0), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
      "Wrong type");
  auto t1 = tlv_dynamic.tensor_view<3>(1);
  static_assert(std::is_same<decltype(t1), TensorView<EmptyBackendTag, int, 3>>::value,
                "Wrong type");
  auto t2 = tlv_dynamic.tensor_view<DynamicDimensions>(2);
  static_assert(
      std::is_same<decltype(t2), TensorView<EmptyBackendTag, int, DynamicDimensions>>::value,
      "Wrong type");
  EXPECT_EQ(t2.dim(), 3);
}

TEST(TensorListViewTest, TypePromotion) {
  TensorListShape<3> shape({ { 1, 2, 3}, {4, 5, 6 } });
  TensorListView<EmptyBackendTag, int, 3> tv{nullptr, shape};
  TensorListView<EmptyBackendTag, const int, 3> tvc = tv;
  EXPECT_EQ(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, tv.data);
  tvc = {};
  EXPECT_NE(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, nullptr);
  tvc = tv;
  EXPECT_EQ(tvc.shape, tv.shape);
  EXPECT_EQ(tvc.data, tv.data);

  TensorListView<EmptyBackendTag, int> tv_dyn = tv;
  EXPECT_EQ(tv_dyn.shape, tv.shape);
  EXPECT_EQ(tv_dyn.data, tv.data);

  TensorListView<EmptyBackendTag, const int> tvc_dyn = tv;
  EXPECT_EQ(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, tv.data);
  tvc_dyn = {};
  EXPECT_NE(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, nullptr);
  tvc_dyn = tv;
  EXPECT_EQ(tvc_dyn.shape, tv.shape);
  EXPECT_EQ(tvc_dyn.data, tv.data);

  auto *ptr = tv_dyn.shape.shapes.data();
  tvc_dyn = std::move(tv_dyn);
  EXPECT_EQ(tvc_dyn.shape.shapes.data(), ptr) << "Move is broken - a copy appeared somewhere.";
}

TEST(TensorListView, uniform_list_shape) {
  int N = 11;
  TensorListShape<> dyn =  uniform_list_shape(N, { 640, 480, 3 });
  TensorListShape<3> stat = uniform_list_shape<3>(N, { 640, 480, 3 });

  int size_c[] = { 640, 480, 3 };
  std::array<int64_t, 3> size_a = { 640, 480, 3 };
  TensorShape<3> ref(640, 480, 3);

  TensorListShape<3> infer1 = uniform_list_shape<3>(N, size_c);
  TensorListShape<3> infer2 = uniform_list_shape<3>(N, size_a);
  EXPECT_EQ(dyn.num_samples(), N);
  for (int i = 0; i < dyn.num_samples(); i++) {
    EXPECT_EQ(dyn.tensor_shape(i), ref);
  }
  EXPECT_EQ(stat, dyn);
  EXPECT_EQ(infer1, dyn);
  EXPECT_EQ(infer2, dyn);
}

namespace {

template<typename DataType, typename Iterable>
void VerifySubtensor(const DataType *data, Iterable dims, int idx) {
  auto subtensor_volume = volume(dims.begin() + 1, dims.end());
  for (int i = 0; i < subtensor_volume; i++) {
    EXPECT_EQ(idx * subtensor_volume + i, data[i]) << "Failed at idx: " << idx << " offset " << i;
  }
}

}  // namespace


TEST(TensorViewTest, StaticSubtensorTest) {
  using namespace std;  // NOLINT
  constexpr size_t kNDims = 4;
  array<int64_t, kNDims> dims = {4, 1, 2, 3};
  vector<int> data(volume(dims), 0);
  iota(data.begin(), data.end(), 0);
  auto tv = make_tensor_cpu<kNDims>(data.data(), dims);
  for (int i = 0; i < dims[0]; i++) {
    auto ret = subtensor(tv, i);
    VerifySubtensor(ret.data, dims, i);
  }
}


TEST(TensorViewTest, DynamicSubtensorTest) {
  using namespace std;  // NOLINT
  vector<int64_t> dims = {4, 2, 1, 2, 3};
  vector<int> data(volume(dims), 0);
  iota(data.begin(), data.end(), 0);
  auto tv = make_tensor_cpu<-1>(data.data(), dims);
  for (int i = 0; i < dims[0]; i++) {
    auto ret = subtensor(tv, i);
    VerifySubtensor(ret.data, dims, i);
  }
}

}  // namespace kernels
}  // namespace dali
