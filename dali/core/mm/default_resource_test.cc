// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <thread>
#include "dali/core/mm/default_resources.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/mm/malloc_resource.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_error.h"
#include "dali/core/device_guard.h"
#include "dali/core/dev_buffer.h"

namespace dali {
namespace mm {

void _Test_FreeDeviceResources();

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
  DeviceBuffer<char> dev;
  dev.resize(1000);
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
}


TEST(MMDefaultResource, GetResource_Managed) {
  auto *rsrc = GetDefaultResource<memory_kind::managed>();
  ASSERT_NE(rsrc, nullptr);

  CUDAStream stream = CUDAStream::Create(true);
  char *mem = nullptr;
  constexpr int size = 1000;
  try {
    mem = static_cast<char*>(rsrc->allocate(size, 32));
  } catch (const CUDAError &e) {
    if ((e.is_drv_api() && e.drv_error() == CUDA_ERROR_NOT_SUPPORTED) ||
        (e.is_rt_api() && e.rt_error() == cudaErrorNotSupported)) {
      GTEST_SKIP() << "Unified memory not supported on this platform";
    }
  }
  ASSERT_NE(mem, nullptr);
  EXPECT_TRUE(mm::detail::is_aligned(mem, 32));
  CUDA_CALL(cudaStreamAttachMemAsync(stream, mem, 0, cudaMemAttachHost));
  for (int i = 0; i < size; i++)
    mem[i] = i + 42;
  char back_copy[size] = {};
  CUDA_CALL(cudaStreamAttachMemAsync(stream, mem, 0, cudaMemAttachSingle));
  CUDA_CALL(cudaMemcpyAsync(back_copy, mem, size, cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamAttachMemAsync(stream, mem, 0, cudaMemAttachHost));
  rsrc->deallocate_async(mem, size, 32, stream_view(stream));
  CUDAEvent event = CUDAEvent::Create();
  CUDA_CALL(cudaEventRecord(event, stream));
  stream = CUDAStream();  // destroy the stream, it should still complete just fine

  for (int i = 0; i < 1000; i++)
    EXPECT_EQ(back_copy[i], static_cast<char>(i + 42));
}

TEST(MMDefaultResource, GetResource_Device) {
  char *stage = nullptr;
  CUDA_CALL(cudaMallocHost(&stage, 2000));
  try {
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
  } catch (...) {
    CUDA_DTOR_CALL(cudaFreeHost(stage));
    throw;
  }
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
      EXPECT_NE(resources[i], nullptr);
      // If we're using plain cudaMalloc, the resource will be the same for all devices
      if (!dynamic_cast<mm::cuda_malloc_memory_resource*>(resources[i])) {
        for (int j = 0; j < i; j++) {
          EXPECT_NE(resources[i], resources[j]) << "Got the same resource for different devices";
        }
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

template <typename Kind, bool async = !std::is_same<Kind, memory_kind::host>::value>
class DummyResource : public memory_resource<Kind> {
  void *do_allocate(size_t, size_t) override {
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) override {
  }
};

template <typename Kind>
class DummyResource<Kind, true> : public async_memory_resource<Kind> {
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

template <typename Kind>
void TestSetDefaultResource() {
  DummyResource<Kind> dummy;
  SetDefaultResource<Kind>(&dummy);
  EXPECT_EQ(GetDefaultResource<Kind>(), &dummy);
  SetDefaultResource<Kind>(nullptr);
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

TEST(MMDefaultResource, InitStampede) {
  vector<std::thread> threads;
  std::atomic_bool f1, f2, stop;
  std::atomic_int cnt;
  f1 = false;
  f2 = false;
  stop = false;
  cnt = 0;
  for (int i = 0; i < 100; i++) {
    threads.emplace_back([&]() {
      while (!stop) {
        while (!f1) {
          if (stop)
            return;
          std::this_thread::yield();
        }
        GetDefaultResource<memory_kind::device>();
        cnt++;
        while (!f2) {
          if (stop)
            return;
          std::this_thread::yield();
        }
        cnt--;
      }
    });
  }
  for (int attempt = 0; attempt < 1000; attempt++) {
    f1 = true;
    while (cnt != static_cast<int>(threads.size()))
      std::this_thread::yield();
    f1 = false;
    SetDefaultResource<memory_kind::device>(nullptr);
    _Test_FreeDeviceResources();
    f2 = true;
    while (cnt != 0)
      std::this_thread::yield();
    f2 = false;
  }
  stop = true;
  for (auto &t : threads)
    t.join();
}

TEST(MMDefaultResource, SetResource_Current) {
  int device_id = 0;
  CUDA_CALL(cudaGetDevice(&device_id));
  DummyResource<memory_kind::device> dummy;
  EXPECT_NO_THROW(SetDefaultDeviceResource(-1, &dummy));
  EXPECT_EQ(GetDefaultDeviceResource(device_id), &dummy);
  EXPECT_EQ(GetDefaultDeviceResource(-1), &dummy);
  SetDefaultDeviceResource(-1, nullptr);
}

TEST(MMDefaultResource, GetResource_Device_RangeCheck) {
  int ndev = 0;
  CUDA_CALL(cudaGetDeviceCount(&ndev));
  EXPECT_NO_THROW(GetDefaultDeviceResource(-1));
  int current_dev;
  CUDA_CALL(cudaGetDevice(&current_dev));
  EXPECT_EQ(GetDefaultDeviceResource(-1), GetDefaultDeviceResource(current_dev));
  for (int i = 0; i < ndev; i++) {
    EXPECT_NO_THROW(GetDefaultDeviceResource(i));
  }
  EXPECT_THROW(GetDefaultDeviceResource(ndev), std::out_of_range);
  EXPECT_THROW(GetDefaultDeviceResource(ndev+100), std::out_of_range);
}

}  // namespace test
}  // namespace mm
}  // namespace dali
