#include "ndll/pipeline/data/caching_allocator.h"

namespace ndll {

namespace {
// Hacked version of cub::Debug from 'cub/util_debug.h'
__host__ __device__ __forceinline__ cudaError_t Debug(cudaError_t error,
    const char* filename, int line) {
  (void)filename;
  (void)line;
#ifndef NDEBUG
  if (error) {
    fprintf(stderr, "CUDA error %d [%s, %d]: %s\n",
        error, filename, line, cudaGetErrorString(error));
    fflush(stderr);
  }
#endif // NDEBUG
  return error;
}
} // namespace

// Debug macro replacement for CubDebug & _CubLog
#define NDLL_CUB_DEBUG(code) Debug((cudaError_t)(code), __FILE__, __LINE__)
#define NDLL_CUB_LOG(format, ...) printf(format, __VA_ARGS__);

cudaError_t CachingDeviceAllocator::SetMaxCachedBytes(size_t max_cached_bytes) {
  // Lock
  mutex.lock();

  if (debug) NDLL_CUB_LOG("Changing max_cached_bytes (%lld -> %lld)\n",
      (long long) this->max_cached_bytes, (long long) max_cached_bytes);

  this->max_cached_bytes = max_cached_bytes;

  // Unlock
  mutex.unlock();

  return cudaSuccess;
}

cudaError_t CachingDeviceAllocator::DeviceAllocate(int device,
    void **d_ptr, size_t bytes, cudaStream_t active_stream) {
  *d_ptr                          = NULL;
  int entrypoint_device           = INVALID_DEVICE_ORDINAL;
  cudaError_t error               = cudaSuccess;

  if (device == INVALID_DEVICE_ORDINAL)
  {
    if (NDLL_CUB_DEBUG(error = cudaGetDevice(&entrypoint_device))) return error;
    device = entrypoint_device;
  }

  // Create a block descriptor for the requested allocation
  bool found = false;
  BlockDescriptor search_key(device);
  search_key.associated_stream = active_stream;
  NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

  if (search_key.bin > max_bin)
  {
    // Bin is greater than our maximum bin: allocate the request
    // exactly and give out-of-bounds bin.  It will not be cached
    // for reuse when returned.
    search_key.bin      = INVALID_BIN;
    search_key.bytes    = bytes;
  }
  else
  {
    // Search for a suitable cached allocation: lock
    mutex.lock();

    if (search_key.bin < min_bin)
    {
      // Bin is less than minimum bin: round up
      search_key.bin      = min_bin;
      search_key.bytes    = min_bin_bytes;
    }

    // Iterate through the range of cached blocks on the same device in the same bin
    CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
    while ((block_itr != cached_blocks.end())
        && (block_itr->device == device)
        && (block_itr->bin == search_key.bin))
    {
      // To prevent races with reusing blocks returned by the host but still
      // in use by the device, only consider cached blocks that are
      // either (from the active stream) or (from an idle stream)
      if ((active_stream == block_itr->associated_stream) ||
          (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady))
      {
        // Reuse existing cache block.  Insert into live blocks.
        found = true;
        search_key = *block_itr;
        search_key.associated_stream = active_stream;
        live_blocks.insert(search_key);

        // Remove from free blocks
        cached_bytes[device].free -= search_key.bytes;
        cached_bytes[device].live += search_key.bytes;

        if (debug) NDLL_CUB_LOG("\tDevice %d reused cached block at %p "
            "(%lld bytes) for stream %lld (previously associated with stream %lld).\n",
            device, search_key.d_ptr, (long long) search_key.bytes,
            (long long) search_key.associated_stream,
            (long long)  block_itr->associated_stream);

        cached_blocks.erase(block_itr);

        break;
      }
      block_itr++;
    }

    // Done searching: unlock
    mutex.unlock();
  }

  // Allocate the block if necessary
  if (!found)
  {
    // Set runtime's current device to specified device (entrypoint may not be set)
    if (device != entrypoint_device)
    {
      if (NDLL_CUB_DEBUG(error = cudaGetDevice(&entrypoint_device))) return error;
      if (NDLL_CUB_DEBUG(error = cudaSetDevice(device))) return error;
    }

    // Attempt to allocate
    if (NDLL_CUB_DEBUG(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))
        == cudaErrorMemoryAllocation)
    {
      // The allocation attempt failed: free all cached blocks on device and retry
      if (debug) NDLL_CUB_LOG("\tDevice %d failed to allocate %lld bytes for stream %lld, "
          "retrying after freeing cached allocations",
          device, (long long) search_key.bytes, (long long) search_key.associated_stream);

      error = cudaSuccess;    // Reset the error we will return
      cudaGetLastError();     // Reset CUDART's error

      // Lock
      mutex.lock();

      // Iterate the range of free blocks on the same device
      BlockDescriptor free_key(device);
      CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

      while ((block_itr != cached_blocks.end()) && (block_itr->device == device))
      {
        // No need to worry about synchronization with the device: cudaFree is
        // blocking and will synchronize across all kernels executing
        // on the current device

        // Free device memory and destroy stream event.
        if (NDLL_CUB_DEBUG(error = cudaFree(block_itr->d_ptr))) break;
        if (NDLL_CUB_DEBUG(error = cudaEventDestroy(block_itr->ready_event))) break;

        // Reduce balance and erase entry
        cached_bytes[device].free -= block_itr->bytes;

        if (debug) NDLL_CUB_LOG("\tDevice %d freed %lld bytes.\n\t\t  %lld available "
            "blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
            device, (long long) block_itr->bytes, (long long) cached_blocks.size(),
            (long long) cached_bytes[device].free, (long long) live_blocks.size(),
            (long long) cached_bytes[device].live);

        cached_blocks.erase(block_itr);

        block_itr++;
      }

      // Unlock
      mutex.unlock();

      // Return under error
      if (error) return error;

      // Try to allocate again
      if (NDLL_CUB_DEBUG(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) return error;
    }

    // Create ready event
    if (NDLL_CUB_DEBUG(error = cudaEventCreateWithFlags(&search_key.ready_event,
                cudaEventDisableTiming)))
      return error;

    // Insert into live blocks
    mutex.lock();
    live_blocks.insert(search_key);
    cached_bytes[device].live += search_key.bytes;
    mutex.unlock();

    if (debug) NDLL_CUB_LOG("\tDevice %d allocated new device block at %p "
        "%lld bytes associated with stream %lld).\n",
        device, search_key.d_ptr, (long long) search_key.bytes,
        (long long) search_key.associated_stream);

    // Attempt to revert back to previous device if necessary
    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
    {
      if (NDLL_CUB_DEBUG(error = cudaSetDevice(entrypoint_device))) return error;
    }
  }

  // Copy device pointer to output parameter
  *d_ptr = search_key.d_ptr;

  if (debug) NDLL_CUB_LOG("\t\t%lld available blocks cached (%lld bytes), "
      "%lld live blocks outstanding(%lld bytes).\n",
      (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
      (long long) live_blocks.size(), (long long) cached_bytes[device].live);

  return error;
}

cudaError_t CachingDeviceAllocator::DeviceFree(int device, void* d_ptr) {
  int entrypoint_device           = INVALID_DEVICE_ORDINAL;
  cudaError_t error               = cudaSuccess;

  if (device == INVALID_DEVICE_ORDINAL)
  {
    if (NDLL_CUB_DEBUG(error = cudaGetDevice(&entrypoint_device)))
      return error;
    device = entrypoint_device;
  }

  // Lock
  mutex.lock();

  // Find corresponding block descriptor
  bool recached = false;
  BlockDescriptor search_key(d_ptr, device);
  BusyBlocks::iterator block_itr = live_blocks.find(search_key);
  if (block_itr != live_blocks.end())
  {
    // Remove from live blocks
    search_key = *block_itr;
    live_blocks.erase(block_itr);
    cached_bytes[device].live -= search_key.bytes;

    // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
    if ((search_key.bin != INVALID_BIN) &&
        (cached_bytes[device].free + search_key.bytes <= max_cached_bytes))
    {
      // Insert returned allocation into free blocks
      recached = true;
      cached_blocks.insert(search_key);
      cached_bytes[device].free += search_key.bytes;

      if (debug) NDLL_CUB_LOG("\tDevice %d returned %lld bytes from associated "
          "stream %lld.\n\t\t %lld available blocks cached (%lld bytes), "
          "%lld live blocks outstanding. (%lld bytes)\n",
          device, (long long) search_key.bytes, (long long) search_key.associated_stream,
          (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
          (long long) live_blocks.size(), (long long) cached_bytes[device].live);
    }
  }

  // Unlock
  mutex.unlock();

  // First set to specified device (entrypoint may not be set)
  if (device != entrypoint_device)
  {
    if (NDLL_CUB_DEBUG(error = cudaGetDevice(&entrypoint_device))) return error;
    if (NDLL_CUB_DEBUG(error = cudaSetDevice(device))) return error;
  }

  if (recached)
  {
    // Insert the ready event in the associated stream (must have current device set properly)
    if (NDLL_CUB_DEBUG(error = cudaEventRecord(search_key.ready_event,
                search_key.associated_stream))) return error;
  }
  else
  {
    // Free the allocation from the runtime and cleanup the event.
    if (NDLL_CUB_DEBUG(error = cudaFree(d_ptr))) return error;
    if (NDLL_CUB_DEBUG(error = cudaEventDestroy(search_key.ready_event))) return error;

    if (debug) NDLL_CUB_LOG("\tDevice %d freed %lld bytes from associated stream "
        "%lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live "
        "blocks (%lld bytes) outstanding.\n",
        device, (long long) search_key.bytes, (long long) search_key.associated_stream,
        (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
        (long long) live_blocks.size(), (long long) cached_bytes[device].live);
  }

  // Reset device
  if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
  {
    if (NDLL_CUB_DEBUG(error = cudaSetDevice(entrypoint_device))) return error;
  }

  return error;
}

cudaError_t CachingDeviceAllocator::FreeAllCached() {
  cudaError_t error         = cudaSuccess;
  int entrypoint_device     = INVALID_DEVICE_ORDINAL;
  int current_device        = INVALID_DEVICE_ORDINAL;

  mutex.lock();

  while (!cached_blocks.empty())
  {
    // Get first block
    CachedBlocks::iterator begin = cached_blocks.begin();

    // Get entry-point device ordinal if necessary
    if (entrypoint_device == INVALID_DEVICE_ORDINAL)
    {
      if (NDLL_CUB_DEBUG(error = cudaGetDevice(&entrypoint_device))) break;
    }

    // Set current device ordinal if necessary
    if (begin->device != current_device)
    {
      if (NDLL_CUB_DEBUG(error = cudaSetDevice(begin->device))) break;
      current_device = begin->device;
    }

    // Free device memory
    if (NDLL_CUB_DEBUG(error = cudaFree(begin->d_ptr))) break;
    if (NDLL_CUB_DEBUG(error = cudaEventDestroy(begin->ready_event))) break;

    // Reduce balance and erase entry
    cached_bytes[current_device].free -= begin->bytes;

    if (debug) NDLL_CUB_LOG("\tDevice %d freed %lld bytes.\n\t\t  %lld available "
        "blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
        current_device, (long long) begin->bytes, (long long) cached_blocks.size(),
        (long long) cached_bytes[current_device].free, (long long) live_blocks.size(),
        (long long) cached_bytes[current_device].live);

    cached_blocks.erase(begin);
  }

  mutex.unlock();

  // Attempt to revert back to entry-point device if necessary
  if (entrypoint_device != INVALID_DEVICE_ORDINAL)
  {
    if (NDLL_CUB_DEBUG(error = cudaSetDevice(entrypoint_device))) return error;
  }

  return error;
}

} // namespace ndll
