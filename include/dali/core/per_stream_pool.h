// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_PER_STREAM_POOL_H_
#define DALI_CORE_PER_STREAM_POOL_H_

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <utility>
#include <mutex>
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_error.h"

namespace dali {

/**
 * @brief Represents a pool of per-device objects which can be leased to perform job on a stream
 *
 * The PerStreamPool class manages objects which can be leased for the purpose of fulfilling some
 * task on a particular device and stream.
 * When a request is made, the caller must specify a CUDA stream (and optionally device, otherwise
 * current device is used) for the task associated with the object. The pool returns a "lease"
 * RAII wrapper. When the wrapper goes out of scope, the object is returned to the pool
 * and will be available for lease again when the stream finishes the work that's been scheduled
 * until the lease ended.
 * Additionally, if `reuse_pending_on_same_stream` flag is set, subsequent requests for an object
 * specifying the same stream will return that object even if the associated work has not yet
 * completed - this is useful for objects which are safe for immediate reuse on the same stream,
 * e.g. device buffers.
 * If there are no available objects for current device, a new object is created and leased
 * immediately - it will be returned to the pool when the lease is over.
 *
 * If reuse of objects on same stream is not wanted, use the alias PerDevicePool, which prevents
 * this behavior.
 *
 * Example:
 * ```
 * class MyClass {
 *  public:
 *   PerStreamPool<GPUWorker> workers;
 *
 *   void DoSomeJob(cudaStream_t stream) {
 *     auto worker = workers.Get(stream);
 *     worker->DoYourJob(stream);
 *     // Here the lease ends and an event is recorded in `stream`.
 *     // When GPU reaches this event, this worker object will be available for reuse
 *     // on other streams for the same CUDA device.
 *   }
 * };
 *
 * ...
 *
 *   MyClass cls;
 *   cls.DoSomeJob(stream1);  // create a new GPUWorker object and use it
 *   cls.DoSomeJob(stream1);  // reuse the same object
 *   cls.DoSomeJob(stream2);  // if the previous job has not finished, create a new GPUWorker
 *   cls.DoSomeJob(stream2);  // reuse the second worker
 *   cudaDeviceSynchronize();
 *   cls.DoSomeJob(stream2);  // associated work is finished, reuses any of the two workers
 * ```
 *
 * Possible future extensions:
 *  - limit the per-device capacity and wait?
 *  - user-provided functor to create new objects, if none are available?
 *
 * @tparam T                              type of the managed object.
 * @tparam mutex_type                     type of the synchronization object used to protect
 *                                        internal structures
 * @tparam reuse_pending_on_same_stream   if true, subsequent requests for an object
 *                                        specifying same stream may return an object associated
 *                                        with an unfinished job on that stream
 */
template <typename T, typename mutex_type = std::mutex, bool reuse_pending_on_same_stream = true>
class PerStreamPool {
 private:
  struct ListNode;
  using ListNodeUPtr = std::unique_ptr<ListNode>;
  struct ListNode {
    T object;
    int device_id = -1;
    cudaStream_t stream = nullptr;
    CUDAEvent event;
    ListNodeUPtr next;

    template <typename... ObjectConstructorArgs>
    ListNode(int device_id, cudaStream_t stream, ObjectConstructorArgs&&... args)
    : object(std::forward<ObjectConstructorArgs>(args)...)
    , device_id(device_id)
    , stream(stream)
    , event(CUDAEvent::Create()) {
    }

    ~ListNode() {
      DeleteNonRecursive(std::move(next));
    }

    static void DeleteNonRecursive(ListNodeUPtr ptr) {
      while (ptr) {
        auto tmp = std::move(ptr->next);
        // ptr will be deleted, but its next pointer is null now, so we'll not recurse
        ptr = std::move(tmp);
      }
    }
  };

 public:
  class ObjectLease {
   public:
    ObjectLease(ObjectLease &&) = default;
    ~ObjectLease() { Release(owner, std::move(node)); }
    operator T*() const { return &node->object; }
    T &operator*() const { return node->object; }
    T *operator->() const { return &node->object; }
    explicit operator bool() const noexcept { return node != nullptr; }
   private:
    ObjectLease(PerStreamPool *owner, ListNodeUPtr node) : owner(owner), node(std::move(node)) {}
    PerStreamPool *owner;
    ListNodeUPtr node;
    friend class PerStreamPool;
  };

  /**
   * @brief Gets an object from the specified device's pool or creates one if none are available
   * @param device_id CUDA device index
   * @param stream CUDA stream
   * @param args Constructor arguments (used when building a new object)
   */
  template <typename... ObjectConstructorArgs>
  ObjectLease Get(int device_id, cudaStream_t stream, ObjectConstructorArgs&&... args) {
    std::lock_guard<mutex_type> guard(lock_);
    RecycleFinished();
    if (reuse_pending_on_same_stream) {
      if (auto ret = GetPending(stream))
        return {this, std::move(ret)};
    }
    return {this,
            GetFromDevicePool(device_id, stream, std::forward<ObjectConstructorArgs>(args)...)};
  }

  /**
   * @brief Gets an object from the current device's pool or creates one if none are available
   * @param stream CUDA stream
   * @param device_id CUDA device index
   * @param args Constructor arguments (used when building a new object)
   */
  template <typename... ObjectConstructorArgs>
  ObjectLease Get(cudaStream_t stream, ObjectConstructorArgs&&... args) {
    int device = -1;
    CUDA_CALL(cudaGetDevice(&device));
    return Get(device, stream, std::forward<ObjectConstructorArgs>(args)...);
  }

 private:
  friend class ObjectLease;

  void Release(std::unique_ptr<ListNode> node) {
    CUDA_DTOR_CALL(cudaEventRecord(node->event, node->stream));
    std::lock_guard<mutex_type> guard(lock_);
    node->next = std::move(pending_);
    pending_ = std::move(node);
  }

  // Workaround for a bug in some compiler versions, unable to call `owner->Release`
  // in ~ObjectLease.
  static inline void Release(PerStreamPool *This, std::unique_ptr<ListNode> node) {
    This->Release(std::move(node));
  }

  ListNodeUPtr GetPending(cudaStream_t stream) {
    for (ListNodeUPtr *pptr = &pending_; *pptr; pptr = &(*pptr)->next) {
      if ((*pptr)->stream == stream) {
        ListNodeUPtr ret = std::move(*pptr);
        *pptr = std::move(ret->next);
        return ret;
      }
    }
    return nullptr;
  }

  template <typename... ObjectConstructorArgs>
  ListNodeUPtr GetFromDevicePool(int device_id, cudaStream_t stream,
                                 ObjectConstructorArgs&&... args) {
    int idx = device_id + 1;  // account for device -1
    if (idx >= static_cast<int>(dev_pools_.size()) || !dev_pools_[idx])
      return std::make_unique<ListNode>(device_id, stream,
                                        std::forward<ObjectConstructorArgs>(args)...);

    ListNodeUPtr ptr = std::move(dev_pools_[idx]);
    dev_pools_[idx] = std::move(ptr->next);
    ptr->stream = stream;
    return ptr;
  }

  void RecycleFinished() {
    for (ListNodeUPtr *pptr = &pending_; *pptr; ) {
      if (cudaErrorNotReady == cudaEventQuery((*pptr)->event)) {  // still pending_? skip!
        pptr = &(*pptr)->next;
      } else {  // otherwise it's finished (or an error; we still recycle)
        int idx = (*pptr)->device_id + 1;
        if (idx >= static_cast<int>(dev_pools_.size()))
          dev_pools_.resize(idx + 1);

        auto to_recycle = std::move(*pptr);  // remove from pending_ list
        *pptr = std::move(to_recycle->next);  // reconnect the pending_ list

        to_recycle->next = std::move(dev_pools_[idx]);  // add at the head...
        dev_pools_[idx] = std::move(to_recycle);  // ..of dev_pools_[idx]
      }
    }
  }

  std::vector<ListNodeUPtr> dev_pools_;
  ListNodeUPtr pending_;

  mutex_type lock_;
};

/**
 * @brief This pre-configured pool object does not reuse objects for which there is outstanding
 *        work, regardless of the stream.
 *
 * The object maintains a separate, lazily populated, pool of objects for each device.
 * When multiple requests are made faster than the associated GPU work is completed, more objects
 * are created and added to the pool.
 * When the associated work is complete, the objects can be reused in subsequent calls to Get, but
 * not before. This is in contrast with PerStreamPool, which allows the objects to be reused
 * on the same stream as soon as the host-side lease is finished.
 *
 * Example:
 * ```
 * class MyClass {
 *  public:
 *   PerDevicePool<GPUWorker> workers;
 *
 *   void DoSomeJob(cudaStream_t stream) {
 *     auto worker = workers.Get(stream);
 *     worker->DoYourJob(stream);
 *     // Here the lease ends and an event is recorded in `stream`.
 *     // When GPU reaches this event, this worker object will be available for reuse
 *     // on other streams for the same CUDA device.
 *   }
 * };
 *
 * ...
 *
 *   MyClass cls;
 *   cls.DoSomeJob(stream1);  // create a new GPUWorker object and use it
 *   cls.DoSomeJob(stream1);  // if the previous job has not finished, create a new GPUWorker
 *   cls.DoSomeJob(stream2);  // if the previous job has not finished, create a new GPUWorker
 *   cls.DoSomeJob(stream2);  // if the previous job has not finished, create a new GPUWorker
 *   cudaDeviceSynchronize();
 *   cls.DoSomeJob(stream2);  // associated work is finished, reuses any of the previously
 *                            // created workers
 * ```
 *
 * @see PerStreamPool for detailed description.
 */
template <typename T, typename mutex_type = std::mutex>
using PerDevicePool = PerStreamPool<T, mutex_type, false>;

}  // namespace dali

#endif  // DALI_CORE_PER_STREAM_POOL_H_
