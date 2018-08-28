// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_BUFFER_MANAGER_H_
#define DALI_PIPELINE_DATA_BUFFER_MANAGER_H_

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/data/allocator_manager.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {

#define LINEAR_SEARCH

class BufferManagerBase {
 public:
  static constexpr double buffer_size_threshold = 2.0;

  explicit BufferManagerBase(int device) : device_(device) {}
  virtual ~BufferManagerBase() = default;

  virtual unique_ptr<Buffer<CPUBackend>> AcquireBuffer(const size_t nbytes, const bool pinned) = 0;
  virtual unique_ptr<Buffer<GPUBackend>> AcquireBuffer(const size_t nbytes) = 0;

  virtual void ReleaseBuffer(unique_ptr<Buffer<CPUBackend>> *buffer, const bool pinned) = 0;
  virtual void ReleaseBuffer(unique_ptr<Buffer<GPUBackend>> *buffer) = 0;

 protected:
  int device_;

  // locks to ensure thread-safety of buffer lists
  std::mutex cpu_buffers_mutex_, pinned_buffers_mutex_, gpu_buffers_mutex_;
};

// Linear search over available buffers to get the smallest available that
// is >= bytes requested
template <typename Backend>
inline unique_ptr<Buffer<Backend>> LinearSearch(vector<unique_ptr<Buffer<Backend>>> *buffers,
                                                size_t nbytes,
                                                bool pinned = false) {
  int current_idx = -1;
  size_t current_delta = (size_t) -1;

  for (size_t i = 0; i < buffers->size(); ++i) {
    const size_t buffer_size = (*buffers)[i]->capacity();
    if (buffer_size >= (1.0/BufferManagerBase::buffer_size_threshold) * nbytes &&
        buffer_size <= BufferManagerBase::buffer_size_threshold * nbytes &&
        (size_t) abs(buffer_size - nbytes) < current_delta) {
      current_idx = i;
      current_delta = abs(buffer_size - nbytes);
    }
  }
  std::unique_ptr<Buffer<Backend>> buff;
  if (current_idx == -1) {
    auto b = new Buffer<Backend>;
    b->set_pinned(pinned);
    buff.reset(b);
  } else {
    // here we have the best fit - get the buffer and remove from vector
    buff = std::move((*buffers)[current_idx]);
    // remove the (now empty) buffer from the pool
    // Note: For some reason using buffers.erase(buffers.begin() + current_idx)
    // causes huge numbers of allocations..
    std::swap((*buffers)[current_idx], buffers->back());
    buffers->pop_back();
  }

  return buff;
}

class LinearBufferManager : public BufferManagerBase {
 public:
  explicit LinearBufferManager(int device) :
    BufferManagerBase(device) {
    std::lock_guard<std::mutex> lock(cpu_buffers_mutex_);
    int original_device;
    CUDA_CALL(cudaGetDevice(&original_device));
    CUDA_CALL(cudaSetDevice(device_));

    const int buffers_init_size = 0;
    const size_t buffer_initial_size = 1048576;

    cpu_buffers_.resize(buffers_init_size);
    pinned_buffers_.resize(buffers_init_size);
    gpu_buffers_.resize(buffers_init_size);

    for (int i = 0; i < buffers_init_size; ++i) {
      cpu_buffers_[i].reset(new Buffer<CPUBackend>);
      cpu_buffers_[i]->Resize({buffer_initial_size});

      pinned_buffers_[i].reset(new Buffer<CPUBackend>);
      pinned_buffers_[i]->set_pinned(true);
      pinned_buffers_[i]->Resize({buffer_initial_size});

      gpu_buffers_[i].reset(new Buffer<GPUBackend>);
      gpu_buffers_[i]->Resize({buffer_initial_size});
    }

    CUDA_CALL(cudaSetDevice(original_device));
  }

  // Acquire a buffer
  inline unique_ptr<Buffer<CPUBackend>> AcquireBuffer(const size_t nbytes,
                                                      const bool pinned) override;
  inline unique_ptr<Buffer<GPUBackend>> AcquireBuffer(const size_t nbytes) override;

  // Release a buffer
  inline void ReleaseBuffer(unique_ptr<Buffer<CPUBackend>> *buffer,
                            const bool pinned) override;
  inline void ReleaseBuffer(unique_ptr<Buffer<GPUBackend>> *buffer) override;

 private:
  // Buffer lists for re-use
  vector<unique_ptr<Buffer<CPUBackend>>> cpu_buffers_;
  vector<unique_ptr<Buffer<GPUBackend>>> gpu_buffers_;
  vector<unique_ptr<Buffer<CPUBackend>>> pinned_buffers_;
};

unique_ptr<Buffer<CPUBackend>> LinearBufferManager::AcquireBuffer(const size_t bytes,
                                                                  const bool pinned) {
  if (pinned) {
    std::lock_guard<std::mutex> lock(pinned_buffers_mutex_);
    return LinearSearch(&pinned_buffers_, bytes, true);
  } else {
    std::lock_guard<std::mutex> lock(cpu_buffers_mutex_);
    return LinearSearch(&cpu_buffers_, bytes);
  }
}

void LinearBufferManager::ReleaseBuffer(unique_ptr<Buffer<CPUBackend>> *buffer, bool pinned) {
  if (pinned) {
    std::lock_guard<std::mutex> lock(pinned_buffers_mutex_);
    pinned_buffers_.push_back(std::move(*buffer));
  } else {
    std::lock_guard<std::mutex> lock(cpu_buffers_mutex_);
    cpu_buffers_.push_back(std::move(*buffer));
  }
}

unique_ptr<Buffer<GPUBackend>> LinearBufferManager::AcquireBuffer(const size_t bytes) {
  std::lock_guard<std::mutex> lock(gpu_buffers_mutex_);
  auto buf = LinearSearch(&gpu_buffers_, bytes);
  return buf;
}

void LinearBufferManager::ReleaseBuffer(unique_ptr<Buffer<GPUBackend>> *buffer) {
  std::lock_guard<std::mutex> lock(gpu_buffers_mutex_);
  gpu_buffers_.push_back(std::move(*buffer));
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_BUFFER_MANAGER_H_
