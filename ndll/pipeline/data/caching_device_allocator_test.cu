/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include "ndll/pipeline/data/caching_device_allocator.h"

#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

namespace {
// NoOp for testing
template <typename T>
__global__ void EmptyKernel(void) {}

} // namespace

class CachingDeviceAllocatorTest : public NDLLTest {
public:
  void SetUp() {
    NDLLTest::SetUp();
    CUDA_CALL(cudaSetDevice(initial_gpu_));
  }
  int initial_gpu_ = 0;
};

// Taken from 'test/test_allocator.cu' in CUB
TEST_F(CachingDeviceAllocatorTest, CUBTests) {
  CachingDeviceAllocator allocator(new GPUAllocator);
#ifndef NDEBUG
  allocator.debug = true;
#endif

  //
  // Test0
  //
  
  // Create a new stream
  cudaStream_t other_stream;
  CUDA_CALL(cudaStreamCreate(&other_stream));

  // Allocate 999 bytes on the current gpu in stream0
  char *d_999B_stream0_a;
  char *d_999B_stream0_b;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));

  // Run some big kernel in stream 0
  EmptyKernel<void><<<32000, 512, 1024 * 8, 0>>>();

  // Free d_999B_stream0_a
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_a));

  // Allocate another 999 bytes in stream 0
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

  // Check that that we have 1 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  // Check that that we have no cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 0u);

  // Run some big kernel in stream 0
  EmptyKernel<void><<<32000, 512, 1024 * 8, 0>>>();

  // Free d_999B_stream0_b
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_b));

  // Allocate 999 bytes on the current gpu in other_stream
  char *d_999B_stream_other_a;
  char *d_999B_stream_other_b;
  allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream);

  // Check that that we have 1 live blocks on the initial GPU (that we allocated a new one because d_999B_stream0_b is only available for stream 0 until it becomes idle)
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  // Check that that we have one cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 1u);

  // Run some big kernel in other_stream
  EmptyKernel<void><<<32000, 512, 1024 * 8, other_stream>>>();

  // Free d_999B_stream_other
  CUDA_CALL(allocator.DeviceFree(d_999B_stream_other_a));

  // Check that we can now use both allocations in stream 0 after synchronizing the device
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

  // Check that that we have 2 live blocks on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 2u);

  // Check that that we have no cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 0u);

  // Free d_999B_stream0_a and d_999B_stream0_b
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_a));
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_b));

  // Check that we can now use both allocations in other_stream
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream));
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream_other_b, 999, other_stream));

  // Check that that we have 2 live blocks on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 2u);

  // Check that that we have no cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 0u);

  // Run some big kernel in other_stream
  EmptyKernel<void><<<32000, 512, 1024 * 8, other_stream>>>();

  // Free d_999B_stream_other_a and d_999B_stream_other_b
  CUDA_CALL(allocator.DeviceFree(d_999B_stream_other_a));
  CUDA_CALL(allocator.DeviceFree(d_999B_stream_other_b));

  // Check that we can now use both allocations in stream 0 after synchronizing the device and destroying the other stream
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaStreamDestroy(other_stream));
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

  // Check that that we have 2 live blocks on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 2u);

  // Check that that we have no cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 0u);

  // Free d_999B_stream0_a and d_999B_stream0_b
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_a));
  CUDA_CALL(allocator.DeviceFree(d_999B_stream0_b));

  // Free all cached
  CUDA_CALL(allocator.FreeAllCached());

  //
  // Test1
  //

  // Allocate 5 bytes on the current gpu
  char *d_5B;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_5B, 5u));

  // Check that that we have zero free bytes cached on the initial GPU
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, 0u);

  // Check that that we have 1 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  //
  // Test2
  //

  // Allocate 4096 bytes on the current gpu
  char *d_4096B;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_4096B, 4096u));

  // Check that that we have 2 live blocks on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 2u);

  //
  // Test3
  //

  // DeviceFree d_5B
  CUDA_CALL(allocator.DeviceFree(d_5B));

  // Check that that we have min_bin_bytes free bytes cached on the initial gpu
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, allocator.min_bin_bytes);

  // Check that that we have 1 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  // Check that that we have 1 cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 1u);

  //
  // Test4
  //

  // DeviceFree d_4096B
  CUDA_CALL(allocator.DeviceFree(d_4096B));

  // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, allocator.min_bin_bytes + 4096);

  // Check that that we have 0 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 0u);

  // Check that that we have 2 cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 2u);

  //
  // Test5
  //

  // Allocate 768 bytes on the current gpu
  char *d_768B;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_768B, 768u));

  // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, allocator.min_bin_bytes);

  // Check that that we have 1 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  // Check that that we have 1 cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 1u);

  //
  // Test6
  //

  // Allocate max_cached_bytes on the current gpu
  char *d_max_cached;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_max_cached, allocator.max_cached_bytes));

  // DeviceFree d_max_cached
  CUDA_CALL(allocator.DeviceFree(d_max_cached));

  // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, allocator.min_bin_bytes);

  // Check that that we have 1 live block on the initial GPU
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  // Check that that we still have 1 cached block on the initial GPU
  ASSERT_EQ(allocator.cached_blocks.size(), 1u);

  //
  // Test7
  //

  // Free all cached blocks on all GPUs
  CUDA_CALL(allocator.FreeAllCached());

  // Check that that we have 0 bytes cached on the initial GPU
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, 0u);

  // Check that that we have 0 cached blocks across all GPUs
  ASSERT_EQ(allocator.cached_blocks.size(), 0u);

  // Check that that still we have 1 live block across all GPUs
  ASSERT_EQ(allocator.live_blocks.size(), 1u);

  //
  // Test8
  //

  // Allocate max cached bytes + 1 on the current gpu
  char *d_max_cached_plus;
  CUDA_CALL(allocator.DeviceAllocate((void **) &d_max_cached_plus, allocator.max_cached_bytes + 1));

  // DeviceFree max cached bytes
  CUDA_CALL(allocator.DeviceFree(d_max_cached_plus));

  // DeviceFree d_768B
  CUDA_CALL(allocator.DeviceFree(d_768B));

  unsigned int power;
  size_t rounded_bytes;
  allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

  // Check that that we have 4096 free bytes cached on the initial gpu
  ASSERT_EQ(allocator.cached_bytes[initial_gpu_].free, rounded_bytes);

  // Check that that we have 1 cached blocks across all GPUs
  ASSERT_EQ(allocator.cached_blocks.size(), 1u);

  // Check that that still we have 0 live block across all GPUs
  ASSERT_EQ(allocator.live_blocks.size(), 0u);
}

} // namespace ndll
