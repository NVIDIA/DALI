// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/workspace/workspace.h"
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>

namespace dali {
namespace test {

namespace {

StorageDevice DeviceOf(const TensorList<CPUBackend> &) {
  return StorageDevice::CPU;
}

StorageDevice DeviceOf(const TensorList<GPUBackend> &) {
  return StorageDevice::GPU;
}

Workspace MakeMixedWorkspace() {
  Workspace ws;
  auto cpu_in = std::make_shared<TensorList<CPUBackend>>();
  cpu_in->Resize(uniform_list_shape(2, {3, 4}), DALI_UINT8);
  cpu_in->SetLayout("HW");
  auto gpu_in = std::make_shared<TensorList<GPUBackend>>();
  gpu_in->Resize(uniform_list_shape(3, {5, 6, 7}), DALI_FLOAT);
  gpu_in->SetLayout("HWC");
  ws.AddInput(std::move(cpu_in));
  ws.AddInput(std::move(gpu_in));
  ws.AddOutput(std::make_shared<TensorList<GPUBackend>>());
  ws.AddOutput(std::make_shared<TensorList<CPUBackend>>());
  return ws;
}

}  // namespace

TEST(WorkspaceTest, VisitInputDispatchesOnBackend) {
  Workspace ws = MakeMixedWorkspace();

  int visited = 0;
  ws.VisitInput(0, [&](auto &input) {
    static_assert(std::is_const_v<std::remove_reference_t<decltype(input)>>);
    EXPECT_EQ(DeviceOf(input), StorageDevice::CPU);
    EXPECT_EQ(input.num_samples(), 2);
    visited++;
  });
  ws.VisitInput(1, [&](auto &input) {
    EXPECT_EQ(DeviceOf(input), StorageDevice::GPU);
    EXPECT_EQ(input.num_samples(), 3);
    visited++;
  });
  EXPECT_EQ(visited, 2);
}

TEST(WorkspaceTest, VisitInputReturnsVisitorResult) {
  Workspace ws = MakeMixedWorkspace();
  auto shape_of = [](auto &input) -> const TensorListShape<> & { return input.shape(); };
  EXPECT_EQ(ws.VisitInput(0, shape_of), uniform_list_shape(2, {3, 4}));
  EXPECT_EQ(ws.VisitInput(1, shape_of), uniform_list_shape(3, {5, 6, 7}));
  EXPECT_EQ(ws.VisitInput(0, [](auto &input) { return input.type(); }), DALI_UINT8);
  EXPECT_EQ(ws.VisitInput(1, [](auto &input) { return input.type(); }), DALI_FLOAT);
}

TEST(WorkspaceTest, VisitOutputDispatchesOnBackend) {
  Workspace ws = MakeMixedWorkspace();
  ws.VisitOutput(0, [](auto &output) {
    static_assert(!std::is_const_v<std::remove_reference_t<decltype(output)>>);
    EXPECT_EQ(DeviceOf(output), StorageDevice::GPU);
    output.Resize(uniform_list_shape(4, {1}), DALI_INT32);
  });
  ws.VisitOutput(1, [](auto &output) {
    EXPECT_EQ(DeviceOf(output), StorageDevice::CPU);
    output.Resize(uniform_list_shape(5, {2}), DALI_INT64);
  });
  EXPECT_EQ(ws.GetOutputBatchSize(0), 4);
  EXPECT_EQ(ws.GetOutputDataType(0), DALI_INT32);
  EXPECT_EQ(ws.GetOutputBatchSize(1), 5);
  EXPECT_EQ(ws.GetOutputDataType(1), DALI_INT64);
}

TEST(WorkspaceTest, SetInputOutputLayout) {
  Workspace ws = MakeMixedWorkspace();
  EXPECT_EQ(ws.GetInputLayout(0), "HW");
  EXPECT_EQ(ws.GetInputLayout(1), "HWC");

  ws.SetInputLayout(0, "WH");
  ws.SetInputLayout(1, {});
  EXPECT_EQ(ws.GetInputLayout(0), "WH");
  EXPECT_EQ(ws.GetInputLayout(1), "");
  EXPECT_EQ(ws.Input<CPUBackend>(0).GetLayout(), "WH");
  EXPECT_EQ(ws.Input<GPUBackend>(1).GetLayout(), "");

  ws.VisitOutput(0, [](auto &output) { output.Resize(uniform_list_shape(1, {1, 1}), DALI_UINT8); });
  ws.VisitOutput(1, [](auto &output) { output.Resize(uniform_list_shape(1, {1, 1}), DALI_UINT8); });
  ws.SetOutputLayout(0, "AB");
  ws.SetOutputLayout(1, "CD");
  EXPECT_EQ(ws.GetOutputLayout(0), "AB");
  EXPECT_EQ(ws.GetOutputLayout(1), "CD");
  EXPECT_EQ(ws.Output<GPUBackend>(0).GetLayout(), "AB");
  EXPECT_EQ(ws.Output<CPUBackend>(1).GetLayout(), "CD");
}

TEST(WorkspaceTest, VisitInvalidIndexThrows) {
  Workspace ws = MakeMixedWorkspace();
  EXPECT_THROW(ws.VisitInput(2, [](auto &) {}), DALIException);
  EXPECT_THROW(ws.VisitOutput(-1, [](auto &) {}), DALIException);
  EXPECT_THROW(ws.SetInputLayout(2, {}), DALIException);
}

}  // namespace test
}  // namespace dali
