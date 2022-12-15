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

#include "dali/pipeline/executor/source_info_propagation.h"
#include <gtest/gtest.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/span.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {
namespace test {


template <typename Backend, typename T>
void SetSourceInfo(TensorList<Backend> &tl, T &&infos) {
  int n = tl.num_samples();
  assert(static_cast<int>(std::size(infos)) >= n);
  for (int i = 0; i < n; i++) {
    tl.SetSourceInfo(i, infos[i]);
  }
}

void SetInputSourceInfo(Workspace &ws, int idx, const std::vector<std::string> &infos) {
  if (ws.InputIsType<CPUBackend>(idx))
    SetSourceInfo(ws.UnsafeMutableInput<CPUBackend>(idx), infos);
  else
    SetSourceInfo(ws.UnsafeMutableInput<GPUBackend>(idx), infos);
}

void SetOutputSourceInfo(Workspace &ws, int idx, const std::vector<std::string> &infos) {
  if (ws.OutputIsType<CPUBackend>(idx))
    SetSourceInfo(ws.Output<CPUBackend>(idx), infos);
  else
    SetSourceInfo(ws.Output<GPUBackend>(idx), infos);
}

template <typename Backend, typename T>
void CheckSourceInfo(const TensorList<Backend> &tl, T &&infos) {
  int n = tl.num_samples();
  assert(static_cast<int>(std::size(infos)) >= n);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(tl.GetMeta(i).GetSourceInfo(), infos[i]);
  }
}

void CheckOutputSourceInfo(const Workspace &ws, int idx, const std::vector<std::string> &infos) {
  if (ws.OutputIsType<CPUBackend>(idx))
    CheckSourceInfo(ws.Output<CPUBackend>(idx), infos);
  else
    CheckSourceInfo(ws.Output<GPUBackend>(idx), infos);
}

struct BufferDesc {
  StorageDevice device;
  int batch_size = 1;
};

void BuildWorkspace(Workspace &ws,
                    span<const BufferDesc> input_descs,
                    span<const BufferDesc> output_descs) {
  for (const BufferDesc &desc : input_descs) {
    if (desc.device == StorageDevice::CPU) {
      ws.AddInput(std::make_shared<TensorList<CPUBackend>>(desc.batch_size));
    } else {
      assert(desc.device == StorageDevice::GPU);
      ws.AddInput(std::make_shared<TensorList<GPUBackend>>(desc.batch_size));
    }
  }
  for (const BufferDesc &desc : output_descs) {
    if (desc.device == StorageDevice::CPU) {
      ws.AddOutput(std::make_shared<TensorList<CPUBackend>>(desc.batch_size));
    } else {
      assert(desc.device == StorageDevice::GPU);
      ws.AddOutput(std::make_shared<TensorList<GPUBackend>>(desc.batch_size));
    }
  }
}

void BuildWorkspace(Workspace &ws,
                    std::vector<BufferDesc> input_descs,
                    std::vector<BufferDesc> output_descs) {
  BuildWorkspace(ws, make_cspan(input_descs), make_cspan(output_descs));
}


constexpr auto CPU = StorageDevice::CPU;
constexpr auto GPU = StorageDevice::GPU;

TEST(SourceInfoPropagationTest, Clear) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 2 } }, { { GPU, 2 } });
  SetOutputSourceInfo(ws, 0, { "asdf", "b.jpg" });
  CheckOutputSourceInfo(ws, 0, { "asdf", "b.jpg" });
  ClearOutputSourceInfo(ws);
  CheckOutputSourceInfo(ws, 0, { "", "" });
}

TEST(SourceInfoPropagationTest, Simple) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 2 } }, { { GPU, 2 } });
  SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
  ASSERT_TRUE(PropagateSourceInfo(ws));
  CheckOutputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
}

TEST(SourceInfoPropagationTest, Multi) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { GPU, 2 }, { CPU, 2 } });
  SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
  SetInputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
  ASSERT_TRUE(PropagateSourceInfo(ws));
  CheckOutputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
  CheckOutputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
}

TEST(SourceInfoPropagationTest, OneInputOnly) {
  for (int input_with_info = 0; input_with_info < 2; input_with_info++) {
    Workspace ws;
    BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { GPU, 2 }, { CPU, 2 }, { GPU, 2 } });
    SetInputSourceInfo(ws, input_with_info, { "a.jpg", "b.jpg" });
    ASSERT_TRUE(PropagateSourceInfo(ws));
    CheckOutputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
    CheckOutputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
    CheckOutputSourceInfo(ws, 2, { "a.jpg", "b.jpg" });
  }
}

TEST(SourceInfoPropagationTest, InterleavedBlanks) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { GPU, 2 }, { CPU, 2 }, { GPU, 2 } });
  SetInputSourceInfo(ws, 0, { "a.jpg", "" });
  SetInputSourceInfo(ws, 1, { "", "b.jpg" });
  ASSERT_TRUE(PropagateSourceInfo(ws));
  CheckOutputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
  CheckOutputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
  CheckOutputSourceInfo(ws, 2, { "a.jpg", "b.jpg" });
}

TEST(SourceInfoPropagationTest, PartiallyBlankOutput) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 4 }, { CPU, 4 }, { GPU, 4 } }, { { CPU, 4 } });
  SetInputSourceInfo(ws, 0, { "a.jpg", "", "", "" });
  SetInputSourceInfo(ws, 1, { "", "", "", "d.jpg" });
  SetInputSourceInfo(ws, 2, { "", "b.jpg", "", "" });
  ASSERT_TRUE(PropagateSourceInfo(ws));
  CheckOutputSourceInfo(ws, 0, { "a.jpg", "b.jpg", "", "d.jpg" });
}

TEST(SourceInfoPropagationTest, Error_AlreadySet) {
  Workspace ws;
  BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { CPU, 2 } });
  SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
  SetInputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
  SetOutputSourceInfo(ws, 0, { "", "b.jpg" });
  ASSERT_FALSE(PropagateSourceInfo(ws));
  CheckOutputSourceInfo(ws, 0, { "", "b.jpg" });
}

TEST(SourceInfoPropagationTest, Error_DifferentBatchSizes) {
  {
    Workspace ws;
    BuildWorkspace(ws, { { GPU, 3 }, { CPU, 2 } }, { { CPU, 2 } });
    SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg", "c.jpg" });
    SetInputSourceInfo(ws, 1, { "a.jpg", "b.jpg" });
    ASSERT_FALSE(PropagateSourceInfo(ws));
  }
  {
    Workspace ws;
    BuildWorkspace(ws, { { GPU, 2 } }, { { CPU, 3 } });
    SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
    ASSERT_FALSE(PropagateSourceInfo(ws));
  }
}

TEST(SourceInfoPropagationTest, Error_Clash) {
  {
    Workspace ws;
    BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { CPU, 2 } });
    SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
    SetInputSourceInfo(ws, 1, { "a.jpg", "c.jpg" });
    ASSERT_FALSE(PropagateSourceInfo(ws));
  }
  {
    Workspace ws;
    BuildWorkspace(ws, { { GPU, 2 }, { CPU, 2 } }, { { CPU, 2 } });
    SetInputSourceInfo(ws, 0, { "a.jpg", "b.jpg" });
    SetInputSourceInfo(ws, 1, { "", "c.jpg" });
    ASSERT_FALSE(PropagateSourceInfo(ws));
  }
}

}  // namespace test
}  // namespace dali

