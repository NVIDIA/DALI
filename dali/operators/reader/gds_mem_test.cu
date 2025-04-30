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
#include <fcntl.h>
#include <utility>
#include <vector>
#include <thread>
#include "dali/operators/reader/gds_mem.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/dev_buffer.h"
#include "dali/test/device_test.h"
#include "dali/core/dynlink_cufile.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif  // NVML_ENABLED

namespace dali {
namespace gds {
namespace test {

struct CUFileDriverScope {
  CUFileDriverScope() {
    // cuFileDriverOpen in some versions of cuFile library, can close stdin
    // returning 0 file descriptor to the pool, then dali gets it from the OS opening a file
    // and passing to GDS which cannot handle it properly leading to an error
    int stdin_backup = dup(STDIN_FILENO);
    if (stdin_backup == -1) {
      std::cerr << "dup failed: " << strerror(errno) << "\n";
    }
    CUDA_CALL(cuFileDriverOpen());
    if (stdin_backup != -1) {
      if (fcntl(STDIN_FILENO, F_GETFL) == -1 && errno == EBADF) {
        // Restore stdin from backup
        if (dup2(stdin_backup, STDIN_FILENO) == -1) {
          std::cerr << "dup2 failed: " << strerror(errno) << "\n";
        }
      }
      close(stdin_backup);  // Cleanup backup
    }
  }
  ~CUFileDriverScope() {
    // Here we're the sole owner of cuFile library - we can (and want to) call cuFileDriverClose
    // regardless of the symbol version - but if we compiled with a newer API, then
    // cuFileDriverClose will be redirected to cuFileDriverClose_v2, so we need to
    // punch through that and use the old variant, if the new one is not available.
#ifdef cuFileDriverClose
  #pragma push_macro("cuFileDriverClose")
  #undef cuFileDriverClose
    if (cuFileIsSymbolAvailable("cuFileDriverClose_v2")) {
      // cuFileDriverOpen in some versions of cuFile library, can close stdin
      // returning 0 file descriptor to the pool, then dali gets it from the OS opening a file
      // and passing to GDS which cannot handle it properly leading to an error
      int stdin_backup = dup(STDIN_FILENO);
      if (stdin_backup == -1) {
        std::cerr << "dup failed: " << strerror(errno) << "\n";
      }
      CUDA_CALL(cuFileDriverClose_v2());  // termination on exception is expected
      if (stdin_backup != -1) {
        if (fcntl(STDIN_FILENO, F_GETFL) == -1 && errno == EBADF) {
          // Restore stdin from backup
          if (dup2(stdin_backup, STDIN_FILENO) == -1) {
            std::cerr << "dup2 failed: " << strerror(errno) << "\n";
          }
        }
        close(stdin_backup);  // Cleanup backup
      }
    } else {
      // cuFileDriverOpen is some versions of cuFile library, can close stdin
      // returning 0 file descriptor to the pool, then dali gets it from the OS opening a file
      // and passing to GDS which cannot handle it properly leading to an error
      int stdin_backup = dup(STDIN_FILENO);
      if (stdin_backup == -1) {
        std::cerr << "dup failed: " << strerror(errno) << "\n";
      }
      // we've undefined cuFileDriverClose, so it's no longer redirecting to a versioned symbol
      CUDA_CALL(cuFileDriverClose());  // termination on exception is expected
      if (stdin_backup != -1) {
        if (fcntl(STDIN_FILENO, F_GETFL) == -1 && errno == EBADF) {
          // Restore stdin from backup
          if (dup2(stdin_backup, STDIN_FILENO) == -1) {
            std::cerr << "dup2 failed: " << strerror(errno) << "\n";
          }
        }
        close(stdin_backup);  // Cleanup backup
      }
    }
  #pragma pop_macro("cuFileDriverClose")
#else
    // Compiled with the old API - just call cuFileDriverClose.
    CUDA_CALL(cuFileDriverClose());
#endif
  }
};

template <typename TestBody>
void SkipIfIncompatible(TestBody &&body) {
  // skip test for aarch64 and CUDA < 12.2
#if NVML_ENABLED
  static const int driverVersion = []() {
    auto nvml_handle = nvml::NvmlInstance::CreateNvmlInstance();
    auto ret = nvml::GetCudaDriverVersion();
    return ret;
  }();
#if defined(__aarch64__)
  if (driverVersion < 12020) {
    return;
  }
#endif  // __aarch64__
#endif  // NVML_ENABLED
  try {
    body();
  } catch (const CUFileError &e) {
    if (e.result().err == CU_FILE_PLATFORM_NOT_SUPPORTED ||
        e.result().err == CU_FILE_DEVICE_NOT_SUPPORTED) {
      GTEST_SKIP() << "No GDS-capable device found";
    } else {
      throw;
    }
  }
}

TEST(GDSMem, AllocatorMultiGPU) {
  SkipIfIncompatible([&]{
    CUFileDriverScope scope;
    int ndev;
    CUDA_CALL(cudaGetDeviceCount(&ndev));
    if (ndev < 2) {
      GTEST_SKIP() << "This test requires more than one CUDA capable device to run.";
      return;
    }
    ASSERT_NE(GDSAllocator::get(0), GDSAllocator::get(1));
  });
}

TEST(GDSMem, Allocator) {
  SkipIfIncompatible([&]{
    CUFileDriverScope scope;
    auto alloc = GDSAllocator::get();
    auto unq = alloc->alloc_unique(1024);
    ASSERT_NE(unq, nullptr);
    CUDA_CALL(cudaMemset(unq.get(), 0, 1024));
    unq.reset();
    alloc.reset();
  });
}

TEST(GDSMem, StagingEngine) {
  SkipIfIncompatible([&]{
    CUFileDriverScope scope;
    CUDAStream stream = CUDAStream::Create(true);
    GDSStagingEngine engn;
    size_t chunk = engn.chunk_size();
    auto target_buf = mm::alloc_raw_unique<uint8_t, mm::memory_kind::device>(chunk * 2);
    CUDA_CALL(cudaMemset(target_buf.get(), 0xcc, chunk * 2));
    CUDA_CALL(cudaDeviceSynchronize());
    engn.set_stream(stream);
    auto buf0 = engn.get_staging_buffer();
    auto buf1 = engn.get_staging_buffer();
    engn.return_unused(std::move(buf0));
    auto buf2 = engn.get_staging_buffer(buf1.at(chunk));
    CUDA_CALL(cudaMemsetAsync(buf1.at(0), 0xaa, chunk, stream));
    CUDA_CALL(cudaMemsetAsync(buf2.at(0), 0xbb, chunk, stream));
    engn.copy_to_client(target_buf.get(), chunk, std::move(buf1), 0);
    engn.copy_to_client(target_buf.get() + chunk, chunk, std::move(buf2), 0);
    engn.commit();
    std::vector<uint8_t> host(chunk * 2);
    CUDA_CALL(cudaMemcpyAsync(host.data(), target_buf.get(), 2 * chunk,
                              cudaMemcpyDeviceToHost, stream));
    for (size_t i = 0; i < chunk; i++)
      ASSERT_EQ(host[i], 0xaa);
    for (size_t i = chunk; i < 2*chunk; i++)
      ASSERT_EQ(host[i], 0xbb);
  });
}

namespace {
__global__ void fill_kernel(int *target, int size, int start) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    target[idx] = idx + start;
}

void fill(int *target, int size, int start, cudaStream_t stream) {
  int block = std::min(512, size);
  int grid = div_ceil(size, block);
  fill_kernel<<<grid, block, 0, stream>>>(target, size, start);
}

}  // namespace


DEFINE_TEST_KERNEL(GDSMem, StagingEngineBigTest, const int *target, int size, int start) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    DEV_ASSERT_EQ(target[idx], idx + start);
}

TEST(GDSMem, StagingEngineBigTest) {
  SkipIfIncompatible([&]{
    CUFileDriverScope scope;
    CUDAStream stream = CUDAStream::Create(true);
    GDSStagingEngine engn;
    engn.set_stream(stream);
    const int num_threads = 50;
    std::vector<std::thread> threads;
    int elems_per_thread = 30000000;
    const int chunk = engn.chunk_size();
    const int elems_per_chunk = chunk / sizeof(int);

    DeviceBuffer<int> out[num_threads];

    for (int t = 0; t < num_threads; t++)
      out[t].resize(elems_per_thread);

    for (int t = 0; t < num_threads; t++) {
      threads.emplace_back([&, t]() {
        CUDAStreamLease fill_stream = CUDAStreamPool::instance().Get();
        int elems_written = 0;
        int *prev = nullptr;
        while (elems_written < elems_per_thread) {
          auto buf = engn.get_staging_buffer(prev + elems_per_chunk);
          int n = std::min(elems_per_chunk, elems_per_thread - elems_written);
          fill(static_cast<int*>(buf.at(0)), n, elems_written, fill_stream);
          CUDA_CALL(cudaStreamSynchronize(fill_stream));
          engn.copy_to_client(out[t].data() + elems_written, n * sizeof(int), std::move(buf), 0);
          elems_written += n;
        }
      });
    }
    for (auto &t : threads)
      t.join();
    engn.commit();
    CUDA_CALL(cudaStreamSynchronize(stream));
    for (int t = 0; t < num_threads; t++) {
      dim3 block(1024);
      dim3 grid(div_ceil(elems_per_thread, block.x));
      DEVICE_TEST_CASE_BODY(GDSMem, StagingEngineBigTest, grid, block,
                            out[t].data(), 0, elems_per_thread);
    }
  });
}

}  // namespace test
}  // namespace gds
}  // namespace dali
