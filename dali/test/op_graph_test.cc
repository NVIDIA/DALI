// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <dali/test/op_graph.h>

namespace dali {
namespace testing {

TEST(OpGraph, ValidDAG1) {
  OpDAG dag;
  auto &n1 = dag.add("MyOp1")(dag.in[0]);
  auto &n2 = dag.add("MyOp2")(dag.in[1]);
  auto &n3 = dag.add("MyOp3")(n1[0], n2[0]);
  dag.out(n3[0]);
  EXPECT_TRUE(dag.validate());
}

TEST(OpGraph, ValidDAG2) {
  OpDAG dag;
  auto &n1 = dag.add("MyOp1")(dag.in[0]);
  auto &n2 = dag.add("MyOp2")(dag.in[1], n1[0]);
  auto &n3 = dag.add("MyOp3")(n1[0], n2[0]);
  dag.out(n3[0]);
  EXPECT_TRUE(dag.validate());
}

TEST(OpGraph, InvalidDAG) {
  OpDAG dag;
  auto &n1 = dag.add("MyOp1")(dag.in[0]);
  auto &n2 = dag.add("MyOp2")(dag.in[1], n1[0]);
  auto &n3 = dag.add("MyOp3")(n1[0], n2[0]);
  n2.add_in(n3[0]);
  dag.out(n3[0]);
  EXPECT_FALSE(dag.validate());
}

}  // namespace testing
}  // namespace dali
