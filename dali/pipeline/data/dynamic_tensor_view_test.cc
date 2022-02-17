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
#include "dali/pipeline/data/dynamic_tensor_view.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {

using EBT = EmptyBackendTag;  // allow most of the checks to fit in one line

template <typename DynamicTV, int ndim>
void compare(const DynamicTV &dtv, const void *ptr, const TensorShape<ndim> &shape,
             DALIDataType dtype) {
  EXPECT_EQ(dtv.data, ptr);
  EXPECT_EQ(dtv.shape, shape);
  EXPECT_EQ(dtv.type_id, dtype);
}

template <typename TV>
std::enable_if_t<std::is_same<std::remove_const_t<typename TV::element_type>, void>::value,
                 DALIDataType>
GetDataType(const TV &tv) {
  return tv.type_id;
}

template <typename TV>
std::enable_if_t<!std::is_same<std::remove_const_t<typename TV::element_type>, void>::value,
                 DALIDataType>
GetDataType(const TV &tv) {
  return TypeTable::GetTypeId<typename TV::element_type>();
}

template <typename Left, typename Right>
void compare(const Left &left, const Right &right) {
  EXPECT_EQ(left.data, right.data);
  EXPECT_EQ(left.shape, right.shape);
  EXPECT_EQ(GetDataType(left), GetDataType(right));
}

template <typename Expected, typename Actual, typename Source>
void compare_typed(const Actual &converted, const Source &source) {
  static_assert(std::is_same<Expected, Actual>::value, "Static type test");
  compare(converted, source);
}

TEST(DynamicTypesTensorViewTest, DefaultConstructors) {
  TensorView<EBT, DynamicType, 4> empty_static_dim{};
  compare(empty_static_dim, nullptr, TensorShape<4>{}, DALI_NO_TYPE);

  TensorView<EBT, DynamicType> empty_dynamic_dim{};
  compare(empty_dynamic_dim, nullptr, TensorShape<>{}, DALI_NO_TYPE);

  TensorView<EBT, const DynamicType, 4> const_empty_static_dim{};
  compare(const_empty_static_dim, nullptr, TensorShape<4>{}, DALI_NO_TYPE);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim{};
  compare(const_empty_dynamic_dim, nullptr, TensorShape<>{}, DALI_NO_TYPE);
}

TEST(DynamicTypesTensorViewTest, PtrConstructors) {
  int data = {};
  TensorView<EBT, DynamicType, 4> empty_static_dim{&data, {1, 2, 3, 4}};
  compare(empty_static_dim, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, DynamicType, 4> empty_static_dim_conv{&data, TensorShape<>{1, 2, 3, 4}};
  compare(empty_static_dim_conv, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, DynamicType> empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(empty_dynamic_dim, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, DynamicType> empty_dynamic_dim_conv{&data, TensorShape<4>{1, 2, 3, 4}};
  compare(empty_dynamic_dim_conv, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);
}

TEST(ConstDynamicTypesTensorViewTest, PtrConstructors) {
  int data = {};
  TensorView<EBT, const DynamicType, 4> const_empty_static_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_static_dim, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType, 4> const_empty_static_dim_conv{&data,
                                                                    TensorShape<>{1, 2, 3, 4}};
  compare(const_empty_static_dim_conv, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim_conv{&data,
                                                                  TensorShape<4>{1, 2, 3, 4}};
  compare(const_empty_dynamic_dim_conv, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);
}

TEST(ConstDynamicTypesTensorViewTest, ConstPtrConstructors) {
  const int data = {};
  TensorView<EBT, const DynamicType, 4> const_empty_static_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_static_dim, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType, 4> const_empty_static_dim_conv{&data,
                                                                    TensorShape<>{1, 2, 3, 4}};
  compare(const_empty_static_dim_conv, &data, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim{&data, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim_conv{&data,
                                                                  TensorShape<4>{1, 2, 3, 4}};
  compare(const_empty_dynamic_dim_conv, &data, TensorShape<>{1, 2, 3, 4}, DALI_INT32);
}

TEST(DynamicTypesTensorViewTest, NullPtrConstructors) {
  TensorView<EBT, DynamicType, 4> empty_static_dim{nullptr, {1, 2, 3, 4}};
  compare(empty_static_dim, nullptr, TensorShape<4>{1, 2, 3, 4}, DALI_NO_TYPE);

  TensorView<EBT, DynamicType> empty_dynamic_dim{nullptr, {1, 2, 3, 4}};
  compare(empty_dynamic_dim, nullptr, TensorShape<>{1, 2, 3, 4}, DALI_NO_TYPE);

  TensorView<EBT, const DynamicType, 4> const_empty_static_dim{nullptr, {1, 2, 3, 4}};
  compare(const_empty_static_dim, nullptr, TensorShape<4>{1, 2, 3, 4}, DALI_NO_TYPE);

  TensorView<EBT, const DynamicType> const_empty_dynamic_dim{nullptr, {1, 2, 3, 4}};
  compare(const_empty_dynamic_dim, nullptr, TensorShape<>{1, 2, 3, 4}, DALI_NO_TYPE);
}

TEST(DynamicTypesTensorViewTest, TypeIdConstructors) {
  TensorView<EBT, DynamicType, 4> empty_static_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  compare(empty_static_dim, nullptr, TensorShape<4>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, DynamicType> empty_dynamic_dim{nullptr, {1, 2, 3, 4}, DALI_INT32};
  compare(empty_dynamic_dim, nullptr, TensorShape<>{1, 2, 3, 4}, DALI_INT32);

  TensorView<EBT, DynamicType> empty_dynamic_dim_conv{nullptr, TensorShape<4>{1, 2, 3, 4},
                                                      DALI_INT32};
  compare(empty_dynamic_dim_conv, nullptr, TensorShape<>{1, 2, 3, 4}, DALI_INT32);
}

TEST(DynamicTypesTensorViewTest, ConverterConstructors) {
  int data = {};
  TensorView<EBT, int, 3> tv{&data, {1, 2, 3}};
  TensorView<EBT, int> dyn_tv{&data, {1, 2, 3}};

  TensorView<EBT, DynamicType, 3> static_dim{tv}, static_dim_2{dyn_tv};
  compare(static_dim, tv);
  compare(static_dim_2, dyn_tv);

  TensorView<EBT, DynamicType> dynamic_dim{tv}, dynamic_dim_2{dyn_tv};
  compare(dynamic_dim, tv);
  compare(dynamic_dim_2, dyn_tv);

  TensorView<EBT, DynamicType, 3> copy_static_to_static{static_dim};
  compare(copy_static_to_static, tv);

  TensorView<EBT, DynamicType> copy_static_to_dynamic{static_dim};
  compare(copy_static_to_dynamic, tv);


  TensorView<EBT, DynamicType, 3> copy_dynamic_to_static{dynamic_dim};
  compare(copy_dynamic_to_static, tv);

  TensorView<EBT, DynamicType, 3> copy_dynamic_to_dynamic{dynamic_dim};
  compare(copy_dynamic_to_dynamic, tv);
}

TEST(ConstDynamicTypesTensorViewTest, ConverterConstructors) {
  const int cdata = {};
  TensorView<EBT, const int, 3> ctv{&cdata, {1, 2, 3}};
  TensorView<EBT, const int> dyn_ctv{&cdata, {1, 2, 3}};

  TensorView<EBT, const DynamicType, 3> static_dim{ctv}, static_dim_2{dyn_ctv};
  compare(static_dim, ctv);
  compare(static_dim_2, dyn_ctv);

  TensorView<EBT, const DynamicType> dynamic_dim{ctv}, dynamic_dim_2{dyn_ctv};
  compare(dynamic_dim, ctv);
  compare(dynamic_dim_2, dyn_ctv);

  int data = {};
  TensorView<EBT, int, 3> tv{&data, {1, 2, 3}};
  TensorView<EBT, int> dyn_tv{&data, {1, 2, 3}};

  TensorView<EBT, const DynamicType, 3> static_dim_nonconst{tv}, static_dim_nonconst_2{dyn_tv};
  compare(static_dim_nonconst, tv);
  compare(static_dim_nonconst_2, dyn_tv);

  TensorView<EBT, const DynamicType> dynamic_dim_nonconst{tv}, dynamic_dim_nonconst_2{dyn_tv};
  compare(dynamic_dim_nonconst, tv);
  compare(dynamic_dim_nonconst_2, dyn_tv);

  TensorView<EBT, const DynamicType, 3> copy_static_to_static{static_dim};
  compare(copy_static_to_static, ctv);

  TensorView<EBT, const DynamicType> copy_static_to_dynamic{static_dim};
  compare(copy_static_to_dynamic, ctv);

  TensorView<EBT, const DynamicType, 3> copy_dynamic_to_static{dynamic_dim};
  compare(copy_dynamic_to_static, ctv);

  TensorView<EBT, const DynamicType, 3> copy_dynamic_to_dynamic{dynamic_dim};
  compare(copy_dynamic_to_dynamic, ctv);
}

template <typename Actual>
void test_views_non_const(const Actual &tv) {
  compare_typed<TensorView<EBT, int, 3>>(view<int, 3>(tv), tv);
  compare_typed<TensorView<EBT, int, DynamicDimensions>>(view<int>(tv), tv);
  compare_typed<TensorView<EBT, DynamicType, 3>>(view<DynamicType, 3>(tv), tv);
  compare_typed<TensorView<EBT, DynamicType, DynamicDimensions>>(view<DynamicType>(tv), tv);
}

template <typename Actual>
void test_views_const(const Actual &tv) {
  compare_typed<TensorView<EBT, const int, 3>>(view<const int, 3>(tv), tv);
  compare_typed<TensorView<EBT, const int, DynamicDimensions>>(view<const int>(tv), tv);
  compare_typed<TensorView<EBT, const DynamicType, 3>>(view<const DynamicType, 3>(tv), tv);
  compare_typed<TensorView<EBT, const DynamicType, DynamicDimensions>>(view<const DynamicType>(tv),
                                                                       tv);
}

template <typename Actual>
void test_views(const Actual &tv) {
  test_views_non_const(tv);
  test_views_const(tv);
}

TEST(DynamicTypesTensorViewTest, ViewFunction) {
  auto tv1 = TensorView<EBT, int, 3>{reinterpret_cast<int *>(42), {1, 2, 3}};
  test_views(tv1);

  auto ctv1 = TensorView<EBT, const int, 3>{reinterpret_cast<const int *>(42), {1, 2, 3}};
  test_views_const(ctv1);

  auto tv2 = TensorView<EBT, int>{reinterpret_cast<int *>(42), TensorShape<3>{1, 2, 3}};
  test_views(tv2);

  auto ctv2 =
      TensorView<EBT, const int>{reinterpret_cast<const int *>(42), TensorShape<3>{1, 2, 3}};
  test_views_const(ctv2);

  auto tv3 = TensorView<EBT, DynamicType, 3>{reinterpret_cast<int *>(42), {1, 2, 3}};
  test_views(tv3);

  auto ctv3 = TensorView<EBT, const DynamicType, 3>{reinterpret_cast<const int *>(42), {1, 2, 3}};
  test_views_const(ctv3);

  auto tv4 = TensorView<EBT, DynamicType>{reinterpret_cast<int *>(42), TensorShape<3>{1, 2, 3}};
  test_views(tv4);

  auto ctv4 = TensorView<EBT, const DynamicType>{reinterpret_cast<const int *>(42),
                                                 TensorShape<3>{1, 2, 3}};
  test_views_const(ctv4);
}

TEST(DynamicTypesTensorViewTest, FromTensor) {
  Tensor<CPUBackend> tensor;
  tensor.Resize({1, 2, 3}, DALI_INT32);
  auto dtv = view<DynamicType>(tensor);
  auto cdtv = view<const DynamicType>(tensor);

  auto dtv_3 = view<DynamicType, 3>(tensor);
  auto cdtv_3 = view<const DynamicType, 3>(tensor);

  auto baseline_tv = view<int, 3>(tensor);

  compare(dtv, baseline_tv);
  compare(cdtv, baseline_tv);

  compare(dtv_3, baseline_tv);
  compare(cdtv_3, baseline_tv);
}

}  // namespace dali
