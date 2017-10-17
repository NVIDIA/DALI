#include "ndll/pipeline/data/backend.h"

#include <mutex>

#include "ndll/common.h"
#include "ndll/pipeline/data/allocator.h"
#include "ndll/pipeline/op_spec.h"

namespace ndll {

class AllocatorManager {
public:
  static void SetAllocators(const OpSpec &cpu_allocator, const OpSpec &gpu_allocator) {
    // Lock so we can give a good error if the user calls this from multiple threads.
    std::lock_guard<std::mutex> lock(mutex_);
    NDLL_ENFORCE(cpu_allocator_ == nullptr, "NDLL CPU allocator already set");
    NDLL_ENFORCE(gpu_allocator_ == nullptr, "NDLL GPU allocator already set");
    cpu_allocator_ = CPUAllocatorRegistry::Registry()
      .Create(cpu_allocator.name(), cpu_allocator);
    gpu_allocator_ = GPUAllocatorRegistry::Registry()
      .Create(gpu_allocator.name(), gpu_allocator);
  }
  
  static CPUAllocator& GetCPUAllocator() {
    NDLL_ENFORCE(cpu_allocator_ != nullptr,
        "NDLL CPU allocator not set. Did you forget to call NDLLInit?");
    return *cpu_allocator_.get();
  }
  
  static GPUAllocator& GetGPUAllocator() {
    NDLL_ENFORCE(gpu_allocator_ != nullptr,
        "NDLL GPU allocator not set. Did you forget to call NDLLInit?");
    return *gpu_allocator_.get();
  }
  
private:
  // AllocatorManager should be accessed through its static members
  AllocatorManager() {}
  
  static unique_ptr<CPUAllocator> cpu_allocator_;
  static unique_ptr<GPUAllocator> gpu_allocator_;
  static std::mutex mutex_;
};

unique_ptr<CPUAllocator> AllocatorManager::cpu_allocator_(nullptr);
unique_ptr<GPUAllocator> AllocatorManager::gpu_allocator_(nullptr);
std::mutex AllocatorManager::mutex_;

// Sets the allocator ptrs for all backends
void InitializeBackends(const OpSpec &cpu_allocator,
    const OpSpec &gpu_allocator) {
  AllocatorManager::SetAllocators(cpu_allocator, gpu_allocator);
}

void* GPUBackend::New(size_t bytes) {
  void *ptr = nullptr;
  AllocatorManager::GetGPUAllocator().New(&ptr, bytes);
  return ptr;
}

void GPUBackend::Delete(void *ptr, size_t bytes) {
  AllocatorManager::GetGPUAllocator().Delete(ptr, bytes);
}

void* CPUBackend::New(size_t bytes) {
  void *ptr = nullptr;
  AllocatorManager::GetCPUAllocator().New(&ptr, bytes);
  return ptr;
}

void CPUBackend::Delete(void *ptr, size_t bytes) {
  AllocatorManager::GetCPUAllocator().Delete(ptr, bytes);
}

} // namespace ndll
