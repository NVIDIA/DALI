// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/common.h"  // NOLINT
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

DALI_SCHEMA(PipelineCommonTest).AddOptionalArg("size", "size", std::vector<float>{}, true);

TEST(PipelineCommon, GetShapeLikeArgumentScalar) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  spec.AddArg("size", 1.5f);
  vector<float> shape;
  int D = GetShapeLikeArgument<float>(shape, spec, "size", ws, 3, 5);
  EXPECT_EQ(D, 3);
  ASSERT_EQ(shape.size(), 15);
  for (size_t i = 0; i < 15; i++) {
    EXPECT_EQ(shape[i], 1.5f);
  }
  shape.clear();
}

TEST(PipelineCommon, GetShapeLikeArgumentVector) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  vector<float> src_shape = { -0.75f, 1, 2.75f, 3.25f };
  spec.SetArg("size", src_shape);
  spec.SetArg("batch_size", 3);

  vector<float> shape;
  int D = GetShapeLikeArgument<float>(shape, spec, "size", ws);
  EXPECT_EQ(D, 4);
  ASSERT_EQ(shape.size(), 12);
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 4; d++)
      EXPECT_EQ(shape[i * 4 + d], src_shape[d]);
  }

  vector<int> ref_ishape = { -1, 1, 3, 3 };
  vector<int> ishape;
  spec.SetArg("size", src_shape);
  spec.SetArg("batch_size", 3);
  D = GetShapeLikeArgument<float>(ishape, spec, "size", ws);
  EXPECT_EQ(D, 4);
  ASSERT_EQ(shape.size(), 12);
  for (int i = 0; i < 3; i++) {
    for (int d = 0; d < 4; d++)
      EXPECT_EQ(ishape[i * 4 + d], ref_ishape[d]) << "@ shape[" << i << "][" << d << "]";
  }
}

TEST(PipelineCommon, GetShapeLikeArgumentInput) {
  OpSpec spec("PipelineCommonTest");
  ArgumentWorkspace ws;
  int D = 5;
  int N = 7;
  auto input = std::make_shared<TensorList<CPUBackend>>();
  input->set_pinned(false);

  // specify the shape as a list of 1D tensors
  input->Resize(uniform_list_shape<1>(N, {D}));
  float *shape_data = input->mutable_data<float>();
  for (int i = 0; i < D*N; i++)
    shape_data[i] = i * 1.1f;

  spec.SetArg("batch_size", N);
  spec.AddArgumentInput("size", "size");
  ws.AddArgumentInput("size", input);

  vector<float> shape;

  int out_d = GetShapeLikeArgument<float>(shape, spec, "size", ws, -1, -1);
  EXPECT_EQ(out_d, D) << "Dimensionality should match the size of the tensors in the list.";
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(shape[i * D + d], (i * D + d) * 1.1f);
  }


  // specify the shape as a list of scalars - this will cause the extend to be
  // broadcast to all extents when the extent is known
  ws.Clear();

  input->Resize(TensorListShape<0>(N));
  shape_data = input->mutable_data<float>();
  for (int i = 0; i < N; i++)
    shape_data[i] = i * 1.1f;

  ws.AddArgumentInput("size", input);

  vector<int> ishape;
  out_d = GetShapeLikeArgument<float>(ishape, spec, "size", ws, D, -1);
  EXPECT_EQ(out_d, D) << "A list of scalars can be broadcast to any number of dimensions.";
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(ishape[i * D + d], std::lround(i * 1.1f));
  }

  shape.clear();
  // if the extent is not know, a list of scalars indicates 1D shapes
  out_d = GetShapeLikeArgument<float>(shape, spec, "size", ws, -1, -1);
  EXPECT_EQ(out_d, 1) << "A list of scalars should be interpreted as a 1D shape";
  D = 1;
  ASSERT_EQ(shape.size(), N * D) << "Total size of the shape should be batch x ndim";
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < D; d++)
      EXPECT_EQ(shape[i * D + d], i * 1.1f);
  }
}

}  // namespace dali
