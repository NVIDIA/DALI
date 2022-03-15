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

#ifndef DALI_CORE_ACCESS_ORDER_H_
#define DALI_CORE_ACCESS_ORDER_H_

#include <cuda_runtime.h>
#include <cstddef>
#include "dali/core/api_helper.h"

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

  AccessOrder(cudaStream_t stream);  // NOLINT

  static constexpr AccessOrder host() {
    return AccessOrder(host_sync_stream(), -1);
  }

  /**
   * @brief Returns the underlying handle.
   *
   * Returns the underlying handle. It can be either a genuine CUDA stream or a
   * special host-sync value. When there's no value, the return value is 0, but it should
   * not be treated as a valid CUDA stream.
   */
  cudaStream_t get() const noexcept { return has_value() ? stream_ : 0; }

  /**
   * @brief Returns stream handle, if any.
   *
   * Returns the underlying handle when it denotes a CUDA stream. If this order does not
   * represent a valid CUDA stream, the return value is 0, but it should not be treated
   * as a valid CUDA stream. This behaviour is to facilitate the interoperatibility with
   * legacy functions that take cudaStream_t.
   */
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
    const int i = 4321;
    // C++ standard makes it impossible to have a constexpr pointer with a fixed
    // numerical value, so we have to resort to some compiler specific tricks.
    // Clang and GCC support a __builtin_constant_p hack, other compilers
    // have relaxed constexpr handling.
#ifdef __GNUC__
    return __builtin_constant_p((cudaStream_t)i) ? (cudaStream_t)i : (cudaStream_t)i;
#else
    return static_cast<cudaStream_t>(static_cast<void *>(static_cast<char*>(nullptr) + i));
#endif
  }

  static constexpr cudaStream_t null_stream() noexcept {
    const int i = -1;
    // C++ standard makes it impossible to have a constexpr pointer with a fixed
    // numerical value, so we have to resort to some compiler specific tricks.
    // Clang and GCC support a __builtin_constant_p hack, other compilers
    // have relaxed constexpr handling.
#ifdef __GNUC__
    return __builtin_constant_p((cudaStream_t)i) ? (cudaStream_t)i : (cudaStream_t)i;
#else
    return static_cast<cudaStream_t>(static_cast<void *>(static_cast<char*>(nullptr) + i));
#endif
  }

  /**
   * @brief Waits in `this` ordering context for the work scheduled in the `other` order.
   *
   * Waits for work scheduled in `other` order to complete. If `other` is host, this function
   * is a no-op.
   * If either `other` or `this` is a null order, this function has no effect.
   *
   * @note `wait` is transitive with a notable exception of null streams, for which the funcion
   * is a no-op and can break transitivity.
   *
   * ```
   * AccessOrder o1(stream1), o2(stream2), o3(stream3);
   * o2.wait(o1);
   * o3.wait(o2);  // o3 is now synchronized with o1
   * ```
   * ```
   * AccessOrder o1(stream1), o2(AccessOrder::host()), o3(stream3);
   * o2.wait(o1);  // waits for o1 on host
   * o3.wait(o2);  // no-op, but host is now synchronized with o1, so all new work in o3 will
   *               // be scheduled after o2.wait(o1);
   * ```
   * ```
   * AccessOrder o1(stream1), o2{}, o3(stream3);
   * o2.wait(o1);  // no-op
   * o3.wait(o2);  // no-op - o3 is not synchronized with o1!
   * ```
   */
  void wait(const AccessOrder &other) const;

  /**
   * @brief Waits in `this` ordering context for a CUDA event
   *
   * This function executes `cudaStreamWaitEvent` in case `this` is a stream, or
   * `cudaEventSynchronize` if `this` is host-sync.
   *
   * @note This function is a no-op if `this` is a null order.
   */
  void wait(cudaEvent_t event) const;

  bool operator==(const AccessOrder &other) const noexcept {
    return stream_ == other.stream_ && device_id_ == other.device_id_;
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
