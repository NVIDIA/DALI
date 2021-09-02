// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace {
struct TestStruct {
  int a;
  std::string b;
  float c;
};

struct TestStruct2 {
  int x, y, z;
};


struct TestKernel {
  KernelRequirements Setup(KernelContext &ctx,
                           const InListGPU<float, 3> &in, int arg1, float arg2) {
    ScratchpadEstimator se;
    se.add<mm::memory_kind::device, int>(arg1 + 100);
    KernelRequirements req = {};
    req.output_shapes.push_back(in.shape);
    req.scratch_sizes = se.sizes;
    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<float, 3> &out,
           const InListGPU<float, 3> &in, int arg1, float arg2) {
    EXPECT_NE(ctx.scratchpad->AllocateGPU<int>(arg1+100), nullptr);
    EXPECT_EQ(out.shape, in.shape);
  }
};

}  // namespace

TEST(AnyKernelInstance, CreateOrGet) {
  AnyKernelInstance aki;
  TestStruct &ts1 = aki.create_or_get<TestStruct>(1, "test", 5.0f);
  EXPECT_EQ(ts1.a, 1);
  EXPECT_EQ(ts1.b, "test");
  EXPECT_EQ(ts1.c, 5.0f);
  TestStruct &ts2 = aki.create_or_get<TestStruct>(2, "test2", 6.0f);
  EXPECT_EQ(&ts2, &ts1)  << "Object should not be recreated";
  EXPECT_EQ(ts2.a, 1);
  EXPECT_EQ(ts2.b, "test");
  EXPECT_EQ(ts2.c, 5.0f);
}

TEST(AnyKernelInstance, ChangeType) {
  AnyKernelInstance aki;
  TestStruct &ts1 = aki.create_or_get<TestStruct>(1, "test", 5.0f);
  EXPECT_EQ(ts1.a, 1);
  EXPECT_EQ(ts1.b, "test");
  EXPECT_EQ(ts1.c, 5.0f);
  TestStruct2 &ts2 = aki.create_or_get<TestStruct2>(4, 5, 6);
  EXPECT_EQ(ts2.x, 4);
  EXPECT_EQ(ts2.y, 5);
  EXPECT_EQ(ts2.z, 6);
  TestStruct &ts3 = aki.create_or_get<TestStruct>(9, "xyzw", 7.0f);
  EXPECT_EQ(ts3.a, 9);
  EXPECT_EQ(ts3.b, "xyzw");
  EXPECT_EQ(ts3.c, 7.0f);
}

TEST(AnyKernelInstance, Get) {
  AnyKernelInstance aki;
  EXPECT_THROW(aki.get<TestStruct>(), std::logic_error);
  TestStruct &ts1 = aki.create_or_get<TestStruct>(1, "test", 5.0f);
  TestStruct &ts2 = aki.get<TestStruct>();
  EXPECT_EQ(&ts1, &ts2);

  EXPECT_THROW(aki.get<TestStruct2>(), std::logic_error);
}

TEST(KernelManager, GetScratchpadAllocator) {
  KernelManager mgr;
  mgr.Resize(2, 2);
  ScratchpadAllocator &a0 = mgr.GetScratchpadAllocator(0);
  ScratchpadAllocator &a1 = mgr.GetScratchpadAllocator(1);
  ScratchpadAllocator &a2 = mgr.GetScratchpadAllocator(0);
  EXPECT_NE(&a0, &a1);
  EXPECT_EQ(&a0, &a2);
}

TEST(KernelManager, CreateOrGet_Get) {
  KernelManager mgr;
  mgr.Resize(1, 1);
  auto &k1 = mgr.CreateOrGet<TestKernel>(0);
  auto &k2 = mgr.Get<TestKernel>(0);
  EXPECT_EQ(&k1, &k2);
  EXPECT_THROW(mgr.Get<int>(0), std::logic_error);
  OutListGPU<float, 3> in, out;
  in.resize(2);
  in.shape = {{ { 10, 10, 1 }, { 20, 20, 3 } }};
  out.shape = {{ { 10, 10, 1 }, { 20, 20, 3 } }};
  KernelContext ctx;
  mgr.Setup<TestKernel>(0, ctx, in, 100, 1.25f);
  mgr.Run<TestKernel>(0, 0, ctx, out, in, 100, 1.25f);
}

TEST(KernelManager, TemplateResize) {
  KernelManager mgr;
  mgr.Resize<TestKernel>(1, 1);
  OutListGPU<float, 3> in, out;
  in.resize(2);
  in.shape = {{ { 10, 10, 1 }, { 20, 20, 3 } }};
  out.shape = {{ { 10, 10, 1 }, { 20, 20, 3 } }};
  KernelContext ctx;
  mgr.Setup<TestKernel>(0, ctx, in, 100, 1.25f);
  mgr.Run<TestKernel>(0, 0, ctx, out, in, 100, 1.25f);
}

}  // namespace kernels
}  // namespace dali
