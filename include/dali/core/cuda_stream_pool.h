// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_CUDA_STREAM_POOL_H_
#define DALI_CORE_CUDA_STREAM_POOL_H_

#include <cassert>
#include <atomic>
#include <vector>
#include <utility>
#include "dali/core/cuda_stream.h"
#include "dali/core/spinlock.h"

namespace dali {

class CUDAStreamLease;

class DLL_PUBLIC CUDAStreamPool {
 public:
  CUDAStreamPool();
  ~CUDAStreamPool();

  /**
   * @brief Gets a stream for given device.
   *
   * @param device_id   CUDA runtime API device ordinal. If negative, calling thread's
   *                    current device is used.
   *
   * @return A CUDA stream wrapper object. If there were any streams in the pools, the
   *         stream is taken from it, otherwise a new stream is created.
   */
  CUDAStreamLease Get(int device_id = -1);

  /**
   * @brief Places a stream for given device in the pool.
   *
   * @param device_id CUDA runtime API device ordinal of the device for which the stream was
   *                  created. If negative, the device is obtained from the device context
   *                  associated with the stream.
   *
   * @remarks It is an error to misstate the device_id. Placing a stream with improper device_id
   *          will render the stream pool unusable.
   */
  void Put(CUDAStream &&stream, int device_id = -1);

  /**
   * @brief Removes all streams currently in the pool and deletes auxiliary data structures.
   *
   * NOTE: Avoid using this function on the pool returned by `instance()`
   */
  void Purge();

  /**
   * @brief Returns a reference to the singleton instance.
   *
   * @remarks Using the singleton instance is only possible if:
   * - the enclosing library is compiled and used as a shared object,
   * - the enclosing library is static, but matching calls to Get and Put calls are
   *   contained within one shared object.
   */
  static CUDAStreamPool &instance();

 private:
  friend class CUDAStreamLease;
  friend class CUDAStreamPoolTest;

  void Init();

  CUDAStream GetFromPool(int device_id);

  struct StreamEntry {
    StreamEntry() = default;
    explicit StreamEntry(CUDAStream stream, StreamEntry *next = nullptr)
    : stream(std::move(stream)), next(next) {}
    CUDAStream stream;
    StreamEntry *next = nullptr;
  };

  StreamEntry *unused_ = nullptr;
  std::atomic_int lease_count_{0};

  vector<StreamEntry *> dev_streams_;
  spinlock lock_;

  static StreamEntry *Pop(StreamEntry *&head) {
    auto *e = head;
    if (e)
      head = head->next;
    return e;
  }

  static void Push(StreamEntry *&head, StreamEntry *new_entry) {
    new_entry->next = head;
    head = new_entry;
  }

  void DeleteList(StreamEntry *&head);
};

/**
 * @brief Represents a stream which is leased from a CUDAStreamPool
 *
 * When obtaining a CUDA stream from a CUDAStreamPool, a stream lease object is returned.
 * This object holds a handle to the stream, a device id and a pointer to the stream pool
 * object from which the stream is leased. When the lease is destroyed/reset, the stream is
 * returned to the owning pool.
 *
 * NOTE: The lease must be ended (destroyed or reset) before the owning pool can be destroyed.
 */
class CUDAStreamLease {
 public:
  constexpr CUDAStreamLease() = default;

  CUDAStreamLease(CUDAStreamLease &&other) {
    *this = std::move(other);
  }

  CUDAStreamLease &operator=(CUDAStreamLease &&other) {
    if ((cudaStream_t)stream_ != (cudaStream_t)other.stream_) {  // NOLINT
      stream_ = std::move(other.stream_);
      owner_ = other.owner_;
      device_id_ = other.device_id_;
      other.owner_ = nullptr;
      other.device_id_ = -1;
    }
    return *this;
  }

  ~CUDAStreamLease() {
    reset();
  }

  /**
   * @brief Conversion to a stream handle; enables the use of this object with CUDA runtime APIs.
   */
  operator cudaStream_t() const noexcept {
    return stream_;
  }

  explicit operator bool() const noexcept {
    return stream_;
  }

  /**
   * @brief Detaches the stream from the pool. In most cases this should not be necessary.
   */
  CUDAStream release() noexcept {
    auto ret = std::move(stream_);
    if (owner_) {
      if (ret)
        owner_->lease_count_--;
      owner_ = nullptr;
    }
    device_id_ = -1;
    return ret;
  }

  /**
   * @brief Returns the device ID associated with the stream.
   */
  int device_id() const noexcept {
    return device_id_;
  }

  /**
   * @brief Returns the owning pool object.
   */
  CUDAStreamPool *pool() const noexcept {
    return owner_;
  }

  /**
   * @brief Returns the stream to the owning pool.
   */
  void reset() {
    if (owner_) {
      if (stream_) {
        owner_->Put(std::move(stream_), device_id_);
        owner_->lease_count_--;
      }
      owner_ = nullptr;
    }
    stream_.reset();
    device_id_ = -1;
  }

 private:
  CUDAStreamLease(CUDAStream &&stream, int device_id, CUDAStreamPool *owner)
  : stream_(std::move(stream)), device_id_(device_id), owner_(owner) {
    assert(owner_ && stream_);
    ++owner->lease_count_;
  }

  friend class CUDAStreamPool;
  CUDAStream stream_;
  int device_id_ = -1;
  CUDAStreamPool *owner_ = nullptr;
};


}  // namespace dali

#endif  // DALI_CORE_CUDA_STREAM_POOL_H_
