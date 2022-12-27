// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODE_RESULTS_H_
#define DALI_IMGCODEC_DECODE_RESULTS_H_

#include <stdexcept>
#include <memory>
#include <utility>
#include <vector>
#include "dali/core/api_helper.h"
#include "dali/core/span.h"

namespace dali {
namespace imgcodec {

template <typename T, span_extent_t E = dynamic_extent>
using cspan = span<const T, E>;

/**
 * @brief Results of a decoding operation.
 */
struct DecodeResult {
  bool success = false;
  std::exception_ptr exception = nullptr;

  static DecodeResult Success() { return { true, {} }; }

  static DecodeResult Failure(std::exception_ptr exception) {
    return { false, std::move(exception) };
  }
};


class DecodeResultsSharedState;
class FutureDecodeResults;

/**
 * @brief A promise object for decoding results.
 *
 * When asynchronous decoding is performed, a promise object and copied among the workers.
 * At exit, a future object is obtained from it by a call to get_future.
 * The promise object is what the workers use to notify the caller about the results.
 * The future object is what the caller uses to wait for and access the results.
 */
class DLL_PUBLIC DecodeResultsPromise {
 public:
  explicit DecodeResultsPromise(int num_samples);
  ~DecodeResultsPromise();

  DecodeResultsPromise(const DecodeResultsPromise &other) { *this = other; }
  DecodeResultsPromise(DecodeResultsPromise &&) = default;
  DecodeResultsPromise &operator=(const DecodeResultsPromise &);
  DecodeResultsPromise &operator=(DecodeResultsPromise &&) = default;

  /**
   * @brief Obtains a future object for the caller/consume
   */
  FutureDecodeResults get_future() const;

  /**
   * @brief The number of samples in this promise
   */
  int num_samples() const;

  /**
   * @brief Sets the result for a specific sample
   */
  void set(int index, DecodeResult res);

  /**
   * @brief Sets all results at once
   */
  void set_all(span<DecodeResult> res);

  /**
   * @brief Checks if two promises point to the same shared state.
   */
  bool operator==(const DecodeResultsPromise &other) const {
    return impl_ == other.impl_;
  }

  /**
   * @brief Checks if two promises point to different shared states.
   */
  bool operator!=(const DecodeResultsPromise &other) const {
    return !(*this == other);
  }

 private:
  std::shared_ptr<DecodeResultsSharedState> impl_ = nullptr;
};

/**
 * @brief The object returned by asynchronous decoding requests
 *
 * The future object allows the caller of asynchronous decoding APIs to wait for and obtain
 * partial results, so it can react incrementally to the decoding of mulitple samples,
 * perfomed in the background.
 */
class DLL_PUBLIC FutureDecodeResults {
 public:
  /**
   * @brief Destroys the future object and terminates the program if the results have
   *        not been consumed
   */
  ~FutureDecodeResults();

  FutureDecodeResults(FutureDecodeResults &&other) = default;
  FutureDecodeResults(const FutureDecodeResults &other) = delete;

  FutureDecodeResults &operator=(const FutureDecodeResults &) = delete;
  FutureDecodeResults &operator=(FutureDecodeResults &&other) {
    std::swap(impl_, other.impl_);
    return *this;
  }

  /**
   * @brief Waits for all results to be ready
   */
  void wait_all() const;

  /**
   * @brief Waits for any results that have appeared since the previous call to wait_new
   *        (or any results, if this is the first call).
   *
   * @return The indices of results that are ready. They can be read with `get_one` without waiting.
   */
  cspan<int> wait_new() const;

  /**
   * @brief Waits for the result of a  particualr sample
   */
  void wait_one(int index) const;

  /**
   * @brief The total number of exepcted results.
   */
  int num_samples() const;

  /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
  cspan<DecodeResult> get_all_ref() const;

  /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
  std::vector<DecodeResult> get_all_copy() const;

  /**
   * @brief Waits for all results to become available and returns a span containing the results.
   *
   * @remarks The return value MUST NOT outlive the future object.
   */
  cspan<DecodeResult> get_all() const & {
    return get_all_ref();
  }

  /**
   * @brief Waits for all results to become available and returns a vector containing the results.
   *
   * @remarks The return value is a copy and can outlive the future object.
   */
  std::vector<DecodeResult> get_all() && {
    return get_all_copy();
  }

  /**
   * @brief Waits for a result and returns it.
   */
  DecodeResult get_one(int index) const;

 private:
  explicit FutureDecodeResults(std::shared_ptr<DecodeResultsSharedState> impl);
  friend class DecodeResultsPromise;
  std::shared_ptr<DecodeResultsSharedState> impl_ = nullptr;
};

}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_DECODE_RESULTS_H_

