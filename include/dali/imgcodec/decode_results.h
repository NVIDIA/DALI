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

class DLL_PUBLIC DecodeResultsPromise {
 public:
  explicit DecodeResultsPromise(int num_samples);

  FutureDecodeResults get_future() const;

  int num_samples() const;

  void set(int index, DecodeResult res);

  void set_all(span<DecodeResult> res);

 private:
  std::shared_ptr<DecodeResultsSharedState> impl_ = nullptr;
};

class DLL_PUBLIC FutureDecodeResults {
 public:
  ~FutureDecodeResults();

  FutureDecodeResults(FutureDecodeResults &&other) = default;
  FutureDecodeResults(const FutureDecodeResults &other) = delete;

  FutureDecodeResults &operator=(const FutureDecodeResults &) = delete;
  FutureDecodeResults &operator=(FutureDecodeResults &&other) {
    std::swap(impl_, other.impl_);
    return *this;
  }

  void wait_all() const;

  cspan<int> wait_new() const;

  void wait_one(int index) const;

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

