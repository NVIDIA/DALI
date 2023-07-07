//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STREAM
#define _CUDA_STREAM

#include <cstddef>
// #include <version>
// #include "detail/__cuda_util"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include "dali/core/cuda_error.h"

namespace dali {
namespace cuda_for_dali {

/**
 * \brief A non-owning wrapper for a `cudaStream_t`.
 *
 * `stream_view` is a non-owning "view" type similar to `std::span` or `std::string_view`.
 * \see https://en.cppreference.com/w/cpp/container/span and
 * \see https://en.cppreference.com/w/cpp/string/basic_string_view
 *
 */
class stream_view {
public:

  using value_type = ::cudaStream_t;

  /**
   * \brief Constructs a `stream_view` of the "default" CUDA stream.
   *
   * For behavior of the default stream,
   * \see https://docs.nvidia.com/cuda_for_dali/cuda_for_dali-runtime-api/stream-sync-behavior.html
   *
   */
  constexpr stream_view() noexcept = default;

  /**
   * \brief Constructs a `stream_view` from a `cudaStream_t` handle.
   *
   * This constructor provides implicit conversion from `cudaStream_t`.
   *
   * \note: It is the callers responsibilty to ensure the `stream_view` does not
   * outlive the stream identified by the `cudaStream_t` handle.
   *
   */
  constexpr stream_view(value_type stream) : __stream{stream} {}

  /// Disallow construction from an `int`, e.g., `0`.
  stream_view(int) = delete;

  /// Disallow construction from `nullptr`.
  stream_view(std::nullptr_t) = delete;

  /// Returns the wrapped `cudaStream_t` handle.
  constexpr value_type get() const noexcept { return __stream; }

  /**
   * \brief Synchronizes the wrapped stream.
   *
   * \throws cuda_for_dali::cuda_error if synchronization fails.
   *
   */
  void wait() const {
    CUDA_CALL(::cudaStreamSynchronize(get()));  // "Failed to synchronize stream."
  }

  /**
   * \brief Queries if all operations on the wrapped stream have completed.
   *
   * \throws cuda_for_dali::cuda_error if the query fails.
   *
   * \return `true` if all operations have completed, or `false` if not.
   */
  bool ready() const{
    auto const __result = ::cudaStreamQuery(get());
    if(__result == ::cudaSuccess){
        return true;
    } else if (__result == ::cudaErrorNotReady){
        return false;
    }
    CUDA_CALL(__result);
    return false;
  }

private:
  value_type __stream{0}; ///< Handle of the viewed stream
};

/**
 * \brief Compares two `stream_view`s for equality
 *
 * \note Allows comparison with `cudaStream_t` due to implicit conversion to
 * `stream_view`.
 *
 * \param lhs The first `stream_view` to compare
 * \param rhs The second `stream_view` to compare
 * \return true if equal, false if unequal
 */
inline constexpr bool operator==(stream_view __lhs, stream_view __rhs) {
  return __lhs.get() == __rhs.get();
}

/**
 * \brief Compares two `stream_view`s for inequality
 *
 * \note Allows comparison with `cudaStream_t` due to implicit conversion to
 * `stream_view`.
 *
 * \param lhs The first `stream_view` to compare
 * \param rhs The second `stream_view` to compare
 * \return true if unequal, false if equal
 */
inline constexpr bool operator!=(stream_view __lhs, stream_view __rhs) {
  return not(__lhs == __rhs);
}

}  // namespace cuda_for_dali
}  // namespace dali

#endif //_CUDA_STREAM
