// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/core/mm/default_resources.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"

namespace dali {
namespace mm {
namespace test {

TEST(MMDefaultResource, GetResource_Host) {
  auto *rsrc = GetDefaultResource<memory_kind::host>();
  ASSERT_NE(rsrc, nullptr);
  char *mem = static_cast<char*>(rsrc->allocate(1000, 32));
  ASSERT_NE(mem, nullptr);
  EXPECT_TRUE(mm::detail::is_aligned(mem, 32));
  memset(mem, 42, 1000);
  rsrc->deallocate(mem, 1000, 32);
}

TEST(MMDefaultResource, GetResource_Pinned) {
  char *dev = nullptr;
  CUDA_CALL(cudaMalloc(&dev, 1000));
  CUDA_CALL(cudaMemset(dev, 0, 1000));

  auto *rsrc = GetDefaultResource<memory_kind::pinned>();
  ASSERT_NE(rsrc, nullptr);

  CUDAStream stream = CUDAStream::Create(true);
  char *mem = static_cast<char*>(rsrc->allocate(1000, 32));
  ASSERT_NE(mem, nullptr);
  EXPECT_TRUE(mm::detail::is_aligned(mem, 32));
  for (int i = 0; i < 1000; i++)
    mem[i] = i + 42;
  CUDA_CALL(cudaMemcpyAsync(dev, mem, 1000, cudaMemcpyHostToDevice, stream));
  rsrc->deallocate_async(mem, 1000, 32, stream_view(stream));
  CUDAEvent event = CUDAEvent::Create();
  CUDA_CALL(cudaEventRecord(event, stream));
  stream = CUDAStream();  // destroy the stream, it should still complete just fine
  char back_copy[1000] = {};
  CUDA_CALL(cudaEventSynchronize(event));
  CUDA_CALL(cudaMemcpy(back_copy, dev, 1000, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 1000; i++)
    EXPECT_EQ(back_copy[i], static_cast<char>(i + 42));

  CUDA_CALL(cudaFree(dev));
}

TEST(MMDefaultResource, GetResource_Device) {
  char *stage = nullptr;
  CUDA_CALL(cudaMallocHost(&stage, 2000));
  char *back_copy = stage + 1000;
  memset(back_copy, 0, 10000);
  for (int i = 0; i < 1000; i++)
    stage[i] = i + 42;

  auto *rsrc = GetDefaultResource<memory_kind::device>();
  ASSERT_NE(rsrc, nullptr);

  CUDAStream stream = CUDAStream::Create(true);
  char *mem = static_cast<char*>(rsrc->allocate(1000, 32));
  ASSERT_NE(mem, nullptr);

  EXPECT_TRUE(mm::detail::is_aligned(mem, 32));
  CUDA_CALL(cudaMemcpyAsync(mem, stage, 1000, cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(back_copy, mem, 1000, cudaMemcpyDeviceToHost, stream));
  rsrc->deallocate_async(mem, 1000, 32, stream_view(stream));
  CUDAEvent event = CUDAEvent::Create();
  CUDA_CALL(cudaEventRecord(event, stream));
  stream = CUDAStream();  // destroy the stream, it should still complete just fine
  CUDA_CALL(cudaEventSynchronize(event));

  for (int i = 0; i < 1000; i++)
    EXPECT_EQ(back_copy[i], static_cast<char>(i + 42));

  CUDA_CALL(cudaFreeHost(stage));
}

TEST(MMDefaultResource, GetResource_MultiDevice) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    GTEST_SKIP() << "At least 2 devices needed for the test\n";
  } else {
    DeviceGuard dg;

    vector<async_memory_resource<memory_kind::device>*> resources(ndev, nullptr);
    for (int i = 0; i < ndev; i++) {
      resources[i] = GetDefaultDeviceResource(i);
      for (int j = 0; j < i; j++) {
        EXPECT_NE(resources[i], resources[j]) << "Got the same resource for different devices";
      }
    }

    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      auto *rsrc = GetDefaultResource<memory_kind::device>();
      EXPECT_EQ(rsrc, resources[i]) << "Got different default resource when asked for a specific "
                                       "device than for current device.";
    }
  }
}

template <memory_kind kind, bool async = (kind != memory_kind::host)>
class DummyResource : public memory_resource<kind> {
  void *do_allocate(size_t, size_t) override {
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) override {
  }
};

template <memory_kind kind>
class DummyResource<kind, true> : public async_memory_resource<kind> {
  void *do_allocate(size_t, size_t) override {
    return nullptr;
  }
  void *do_allocate_async(size_t, size_t, stream_view) override {
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) override {
  }
  void do_deallocate_async(void *, size_t, size_t, stream_view) override {
  }
};

template <memory_kind kind>
void TestSetDefaultResource() {
  DummyResource<kind> dummy;
  SetDefaultResource<kind>(&dummy);
  EXPECT_EQ(GetDefaultResource<kind>(), &dummy);
  SetDefaultResource<kind>(nullptr);
}

// TODO(michalz): When memory_kind is a tag type, switch to TYPED_TEST
TEST(MMDefaultResource, SetResource_Host) {
  TestSetDefaultResource<memory_kind::host>();
}

TEST(MMDefaultResource, SetResource_Pinned) {
  TestSetDefaultResource<memory_kind::pinned>();
}

TEST(MMDefaultResource, SetResource_Device) {
  TestSetDefaultResource<memory_kind::device>();
}

TEST(MMDefaultResource, SetResource_MultiDevice) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  if (ndev < 2)
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  std::vector<DummyResource<memory_kind::device>> resources(ndev);
  for (int i = 0; i < ndev; i++) {
    SetDefaultDeviceResource(i, &resources[i]);
    EXPECT_EQ(&resources[i], GetDefaultDeviceResource(i));
    SetDefaultDeviceResource(i, nullptr);
  }
}

}  // namespace test
}  // namespace mm
}  // namespace dali
