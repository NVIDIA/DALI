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


#ifndef DALI_CORE_MM_DETAIL_DEFERRED_DEALLOC_H_
#define DALI_CORE_MM_DETAIL_DEFERRED_DEALLOC_H_

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include "dali/core/mm/detail/util.h"
#include "dali/core/mm/memory_resource.h"
#include "dali/core/small_vector.h"

namespace dali {
namespace mm {

struct dealloc_params {
  int sync_device = -1;  // -1 == current device
  void *ptr = nullptr;
  size_t bytes = 0, alignment = 0;
};

inline bool use_deferred_deallocation(...) {
  return true;
}

/**
 * @brief A resource with deferred deallocation
 *
 * The BaseResource must satisfy the following concept:
 *
 * ```
 * public:
 *   virtual void flush_deferred();
 *   void bulk_deallocate(span<const dealloc_params> deallocs);
 *   bool deferred_dealloc_enabled() const;
 *   int deferred_dealloc_max_outstanding() const;
 * ```
 */
template <typename BaseResource>
class deferred_dealloc_resource : public BaseResource {
 public:
  using base = BaseResource;
  using base::base;

  ~deferred_dealloc_resource() {
    if (worker_.joinable()) {
      stop();
      worker_.join();
      ready_.notify_all();
    }
    this->bulk_deallocate(make_span(deallocs_[0]));
    this->bulk_deallocate(make_span(deallocs_[1]));
  }

  void deferred_deallocate(void *ptr,
                           size_t bytes,
                           size_t alignment = alignof(std::max_align_t),
                           int device_id = -1) {
    if (!ptr || !bytes)
      return;  // nothing to do
    if (device_id < 0) {
      CUDA_CALL(cudaGetDevice(&device_id));
    }

    {
      std::lock_guard<std::mutex> g(mtx_);
      deallocs_[queue_idx_].push_back({device_id, ptr, bytes, alignment});

      if (!started_)
        start_worker();
    }
    cv_.notify_one();
  }


  /**
   * @brief The number of outstanding deferred deallocations
   */
  int deferred_dealloc_count() const {
    return deallocs_[0].size() + deallocs_[1].size();
  }

  /**
   * @brief Waits until currently scheduled deallocations are flushed.
   *
   * This function waits until the worker notifies that it's completed flushing
   * current queue (there are two queues). It doesn't wait for the other queue
   * nor prevent new deallocations from being scheduled.
   *
   * This method overrides the default no-op function from BaseResource.
   */
  void flush_deferred() override {
    if (!no_pending_deallocs()) {
      std::unique_lock<std::mutex> ulock(mtx_);
      if (!no_pending_deallocs() && !stopped_)
        ready_.wait(ulock);
    }
  }

 protected:
  // exposed for testing
  bool no_pending_deallocs() const noexcept {
    return deallocs_[0].empty() && deallocs_[1].empty();
  }

 private:
  using dealloc_queue = SmallVector<dealloc_params, 16>;

  void *do_allocate(size_t bytes, size_t alignment) override {
    if (this->deferred_dealloc_enabled()) {
      if (this->deferred_dealloc_count() > this->deferred_dealloc_max_outstanding())
        this->flush_deferred();
    }
    return base::do_allocate(bytes, alignment);
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    if (this->deferred_dealloc_enabled())
      this->deferred_deallocate(ptr, bytes, alignment, this->device_ordinal());
    else
      base::do_deallocate(ptr, bytes, alignment);
  }


  void start_worker() {
    worker_ = std::thread([this]() {
      run();
    });
    started_ = true;
  }

  void run() {
    int default_device = this->device_ordinal();
    if (default_device >= 0)
      CUDA_CALL(cudaSetDevice(default_device));
    std::unique_lock<std::mutex> ulock(mtx_);
    while (!is_stopped()) {
      cv_.wait(ulock, [&](){ return stopped_ || !deallocs_[queue_idx_].empty(); });
      if (is_stopped())
        break;
      auto &to_free = deallocs_[queue_idx_];
      queue_idx_ = 1 - queue_idx_;
      ulock.unlock();
      this->bulk_deallocate(make_span(to_free));
      ulock.lock();
      to_free.clear();
      ready_.notify_all();
    }
  }

  void stop() {
    stopped_ = true;
    cv_.notify_all();
  }

  bool is_stopped() const noexcept { return stopped_; }

  std::thread worker_;
  std::mutex mtx_;
  std::condition_variable cv_, ready_;
  dealloc_queue deallocs_[2];
  int queue_idx_ = 0;
  bool started_ = false;
  bool stopped_ = false;
};

namespace detail {

template <typename T>
struct can_merge;

template <typename BaseResource>
struct can_merge<deferred_dealloc_resource<BaseResource>> : can_merge<BaseResource> {};

}  // namespace detail

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_DETAIL_DEFERRED_DEALLOC_H_
