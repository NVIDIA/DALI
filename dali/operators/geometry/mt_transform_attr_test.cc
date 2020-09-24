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
#include <memory>
#include <vector>
#include "dali/operators/geometry/coord_transform.h"

namespace dali {

TEST(MTTransformAttr, None) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 3);
  auto M = attr.GetMatrices<3, 3>();
  auto T = attr.GetTranslations<3>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat3::identity());
  for (auto &t : T)
    EXPECT_EQ(t, vec3(0));
}

TEST(MTTransformAttr, MScalarTNone) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{42.0f});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(2);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 2);
  ASSERT_EQ(attr.output_pt_dim_, 2);
  auto M = attr.GetMatrices<2, 2>();
  auto T = attr.GetTranslations<2>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat2(42));
  for (auto &t : T)
    EXPECT_EQ(t, vec2(0));
}

TEST(MTTransformAttr, MScalarTScalar) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{4.0f});
  spec.AddArg("T", vector<float>{5.0f});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 3);
  auto M = attr.GetMatrices<3, 3>();
  auto T = attr.GetTranslations<3>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat3(4));
  for (auto &t : T)
    EXPECT_EQ(t, vec3(5));
}

TEST(MTTransformAttr, MScalarTVector) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{321.0f});
  spec.AddArg("T", vector<float>{1, 2, 3});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 3);
  auto M = attr.GetMatrices<3, 3>();
  auto T = attr.GetTranslations<3>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat3(321));
  for (auto &t : T)
    EXPECT_EQ(t, vec3(1, 2, 3));
}

TEST(MTTransformAttr, MVectorTVector) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{1, 2, 3, 4, 5, 6});
  spec.AddArg("T", vector<float>{10, 20});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 2);
  auto M = attr.GetMatrices<2, 3>();
  auto T = attr.GetTranslations<2>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat2x3({{1, 2, 3}, {4, 5, 6}}) );
  for (auto &t : T)
    EXPECT_EQ(t, vec2(10, 20));
}

TEST(MTTransformAttr, MTVector) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("MT", vector<float>{1, 2, 3, 4, 5, 6});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(2);
  int N = 10;
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 2);
  ASSERT_EQ(attr.output_pt_dim_, 2);
  auto M = attr.GetMatrices<2, 2>();
  auto T = attr.GetTranslations<2>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (auto &m : M)
    EXPECT_EQ(m, mat2({{1, 2}, {4, 5}}) );
  for (auto &t : T)
    EXPECT_EQ(t, vec2(3, 6));
}


TEST(MTTransformAttr, MInputTInput) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto Minp = std::make_shared<TensorVector<CPUBackend>>();
  auto Tinp = std::make_shared<TensorVector<CPUBackend>>();
  Minp->set_pinned(false);
  Tinp->set_pinned(false);
  TensorListShape<2> Mtls = {{{ 2, 3 }, { 2, 3 }}};
  TensorListShape<1> Ttls = {{{ 2 }, { 2 }}};
  int N = Mtls.num_samples();;
  Minp->Resize(Mtls, TypeTable::GetTypeInfo(DALI_FLOAT));
  Tinp->Resize(Ttls, TypeTable::GetTypeInfo(DALI_FLOAT));
  for (int i = 0; i < N; i++) {
    float *data = (*Minp)[i].mutable_data<float>();
    for (int j = 0; j < volume(Mtls[i]); j++)
      data[j] = 1 + j + i * 10;

    data = (*Tinp)[i].mutable_data<float>();
    for (int j = 0; j < volume(Ttls[i]); j++)
      data[j] = 10 + j * 10  + i * 100;
  }

  ws.AddArgumentInput("M", Minp);
  ws.AddArgumentInput("T", Tinp);
  spec.AddArgumentInput("M", "M");
  spec.AddArgumentInput("T", "T");
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 2);
  auto M = attr.GetMatrices<2, 3>();
  auto T = attr.GetTranslations<2>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(M[i], mat2x3({{1, 2, 3}, {4, 5, 6}}) + 10*i);
    EXPECT_EQ(T[i], vec2(10, 20) + 100*i);
  }
}

TEST(MTTransformAttr, MScalarInputTScalarInput) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto Minp = std::make_shared<TensorVector<CPUBackend>>();
  auto Tinp = std::make_shared<TensorVector<CPUBackend>>();
  Minp->set_pinned(false);
  Tinp->set_pinned(false);

  // M and T are inputs containing 0D tensors (scalars)
  TensorListShape<0> tls;
  tls.resize(3);
  int N = tls.num_samples();;
  Minp->Resize(tls, TypeTable::GetTypeInfo(DALI_FLOAT));
  Tinp->Resize(tls, TypeTable::GetTypeInfo(DALI_FLOAT));
  for (int i = 0; i < N; i++) {
    float *data = (*Minp)[i].mutable_data<float>();
    data[0] = i + 10;
    data = (*Tinp)[i].mutable_data<float>();
    data[0] = i + 100;
  }

  ws.AddArgumentInput("M", Minp);
  ws.AddArgumentInput("T", Tinp);
  spec.AddArgumentInput("M", "M");
  spec.AddArgumentInput("T", "T");
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 3);
  ASSERT_EQ(attr.output_pt_dim_, 3);
  auto M = attr.GetMatrices<3, 3>();
  auto T = attr.GetTranslations<3>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(M[i], mat3(i + 10));
    EXPECT_EQ(T[i], vec3(i + 100));
  }
}


TEST(MTTransformAttr, MTInput) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto MTinp = std::make_shared<TensorVector<CPUBackend>>();
  MTinp->set_pinned(false);
  TensorListShape<> tls = {{{ 2, 3 }, { 2, 3 }}};
  int N = tls.num_samples();;
  MTinp->Resize(tls, TypeTable::GetTypeInfo(DALI_FLOAT));
  for (int i = 0; i < N; i++) {
    auto *data = (*MTinp)[i].mutable_data<float>();
    for (int j = 0; j < volume(tls[i]); j++)
      data[j] = 1 + j + i * 10;
  }

  ws.AddArgumentInput("MT", MTinp);
  spec.AddArgumentInput("MT", "MT");
  MTTransformAttr attr(spec);
  attr.SetTransformDims(2);
  attr.ProcessTransformArgs(spec, ws, N);
  ASSERT_EQ(attr.input_pt_dim_, 2);
  ASSERT_EQ(attr.output_pt_dim_, 2);
  auto M = attr.GetMatrices<2, 2>();
  auto T = attr.GetTranslations<2>();
  EXPECT_EQ(M.size(), N);
  EXPECT_EQ(T.size(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(M[i], mat2({{1, 2}, {4, 5}}) + 10*i);
    EXPECT_EQ(T[i], vec2(3, 6) + 10*i);
  }
}


TEST(MTTransformAttr, Error_MScalarTVector) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{321.0f});
  spec.AddArg("T", vector<float>{1, 2, 3, 4});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

TEST(MTTransformAttr, MVectorTVector_ErrorVectorSize) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{1, 2, 3, 4, 5, 6});
  spec.AddArg("T", vector<float>{1, 2, 3});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

TEST(MTTransformAttr, MVector_ErrorEmpty) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

TEST(MTTransformAttr, MVectorTScalar_ErrorNotDivisible) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  spec.AddArg("M", vector<float>{1, 2, 3, 4});
  spec.AddArg("T", vector<float>{1});
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  int N = 10;
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

TEST(MTTransformAttr, MInputTInput_Error) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto Minp = std::make_shared<TensorVector<CPUBackend>>();
  auto Tinp = std::make_shared<TensorVector<CPUBackend>>();
  Minp->set_pinned(false);
  Tinp->set_pinned(false);
  TensorListShape<2> Mtls = {{{ 2, 3 }, { 2, 3 }}};
  TensorListShape<1> Ttls = {{{ 1 }, { 1 }}};
  int N = Mtls.num_samples();;
  Minp->Resize(Mtls, TypeTable::GetTypeInfo(DALI_FLOAT));
  Tinp->Resize(Ttls, TypeTable::GetTypeInfo(DALI_FLOAT));

  {
    ws.AddArgumentInput("M", Minp);
    spec.AddArgumentInput("M", "M");
    MTTransformAttr attr(spec);
    attr.SetTransformDims(2);
    EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);

    attr.SetTransformDims(3);
    EXPECT_NO_THROW(attr.ProcessTransformArgs(spec, ws, N));
  }

  {
    ws.AddArgumentInput("T", Tinp);
    spec.AddArgumentInput("T", "T");

    MTTransformAttr attr(spec);
    attr.SetTransformDims(3);
    EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
  }
}

TEST(MTTransformAttr, MTInput_ErrorSize) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto MTinp = std::make_shared<TensorVector<CPUBackend>>();
  MTinp->set_pinned(false);
  TensorListShape<> tls = {{{ 2, 3 }, { 2, 3 }}};
  int N = tls.num_samples();;
  MTinp->Resize(tls, TypeTable::GetTypeInfo(DALI_FLOAT));
  for (int i = 0; i < N; i++) {
    auto *data = (*MTinp)[i].mutable_data<float>();
    for (int j = 0; j < volume(tls[i]); j++)
      data[j] = 1 + j + i * 10;
  }

  ws.AddArgumentInput("MT", MTinp);
  spec.AddArgumentInput("MT", "MT");
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

TEST(MTTransformAttr, MInput_ZeroRows) {
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto MTinp = std::make_shared<TensorVector<CPUBackend>>();
  MTinp->set_pinned(false);
  TensorListShape<> tls = {{{ 0, 3 }, { 0, 3 }}};
  int N = tls.num_samples();;
  MTinp->Resize(tls, TypeTable::GetTypeInfo(DALI_FLOAT));


  ws.AddArgumentInput("MT", MTinp);
  spec.AddArgumentInput("MT", "MT");
  MTTransformAttr attr(spec);
  attr.SetTransformDims(3);
  EXPECT_THROW(attr.ProcessTransformArgs(spec, ws, N), DALIException);
}

}  // namespace dali
