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

#ifndef DALI_CORE_ACCESS_ORDER_H_
#define DALI_CORE_ACCESS_ORDER_H_

#include <cstddef>
#include "dali/core/api_helper.h"
#include "dali/core/cuda_error.h"
#include "dali/core/cuda_stream.h"

namespace dali {

/**
 * @brief Describes the order in which an object is accessed. It can be stream or host.
 *
 * The access order can be either a CUDA stream order or host order. There's also
 * a special "null" value, useful when passing AccessOrder to functions/objects which may have
 * internal streams.
 */
class DLL_PUBLIC AccessOrder {
 public:
  constexpr AccessOrder() = default;
  constexpr AccessOrder(cudaStream_t stream, int device_id)
  : stream_(stream), device_id_(device_id) {}

  AccessOrder(int) = delete;  // NOLINT  prevent construction from 0
  AccessOrder(std::nullptr_t) = delete;  // NOLINT

  constexpr AccessOrder(cudaStream_t stream) : stream_(stream) {  // NOLINT
    if (is_device())
      device_id_ = DeviceFromStream(stream);
  }

  static constexpr AccessOrder host() {
    return AccessOrder(host_sync_stream());
  }

  cudaStream_t get() const noexcept { return has_value() ? stream_ : 0; }

  cudaStream_t stream() const noexcept { return is_device() ? stream_ : 0; }

  int device_id() const noexcept { return device_id_; }

  bool is_host() const noexcept { return stream_ == host_sync_stream(); }

  bool is_device() const noexcept {
    return stream_ != host_sync_stream() && stream_ != null_stream();
  }

  bool has_value() const noexcept { return stream_ != null_stream(); }

  explicit operator bool() const noexcept { return has_value(); }

  static constexpr cudaStream_t host_sync_stream() noexcept {
    // TODO(michalz): Unify with dali::mm::host_sync
    // Use cast magic and an intermediate variable to make it constexpr
    int i = 4321;
    return static_cast<cudaStream_t>(static_cast<void *>(static_cast<char*>(nullptr) + i));
  }

  static constexpr cudaStream_t null_stream() noexcept {
    // Use cast magic and an intermediate variable to make it constexpr
    int i = -1;
    return static_cast<cudaStream_t>(static_cast<void *>(static_cast<char*>(nullptr) - i));
  }

  /**
   * @brief Waits in `this` ordering context for the work scheduled in the `other` order.
   */
  void join(const AccessOrder &other) const;

  void wait(cudaEvent_t event) const;

  bool operator==(const AccessOrder &other) const noexcept {
    return is_device() == other.is_device() && stream() == other.stream();
  }

  bool operator!=(const AccessOrder &other) const noexcept {
    return !(*this == other);
  }

 private:
  cudaStream_t stream_ = null_stream();
  int device_id_ = -1;
};

}  // namespace dali

#endif  // DALI_CORE_ACCESS_ORDER_H_
