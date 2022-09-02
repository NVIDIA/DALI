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

#include <deque>
#include <mutex>
#include <numeric>
#include <condition_variable>
#include "dali/imgcodec/decode_results.h"

namespace dali {
namespace imgcodec {

class DecodeResultsSharedState {
 public:
  static std::shared_ptr<DecodeResultsSharedState> get() {
    if (free_.empty())
      return std::make_shared<DecodeResultsSharedState>();

    auto ret = std::move(free_.back());
    free_.pop_back();
    return ret;
  }

  static void put(std::shared_ptr<DecodeResultsSharedState> impl) {
    free_.emplace_back(std::move(impl));
  }

  void init(int n) {
    results_.clear();
    results_.resize(n);
    ready_indices_.clear();
    ready_indices_.reserve(n);
    ready_mask_.clear();
    ready_mask_.resize(n);
    last_checked_ = 0;
    has_future_.clear();
  }

  void reset() {
    results_.clear();
    ready_indices_.clear();
    last_checked_ = 0;
    has_future_.clear();
  }

  void wait_all() {
    if (ready_indices_.size() == results_.size())
      return;

    std::unique_lock lock(mtx_);
    cv_any_.wait(lock, [&]() {
      return ready_indices_.size() == results_.size();
    });
  }

  cspan<int> wait_new() {
    if (last_checked_ == results_.size())
      return {};

    std::unique_lock lock(mtx_);
    if (last_checked_ == results_.size())
      return {};

    cv_any_.wait(lock, [&]() {
      return ready_indices_.size() > last_checked_;
    });
    size_t last = last_checked_;
    last_checked_ = ready_indices_.size();
    return make_span(&ready_indices_[last], last_checked_ - last);
  }

  void wait_one(int index) {
    if (!ready_mask_[index]) {
      std::unique_lock lock(mtx_);
      cv_any_.wait(lock, [&]() {
        return ready_mask_[index];
      });
    }
  }

  void set(int index, DecodeResult res) {
    if (static_cast<size_t>(index) >= results_.size())
      throw std::out_of_range("Sample index out of range.");

    std::lock_guard lg(mtx_);
    if (ready_mask_[index])
      throw std::logic_error("Entry already set.");
    results_[index] = std::move(res);
    ready_indices_.push_back(index);
    ready_mask_[index] = true;
    cv_any_.notify_all();
  }

  void set_all(span<DecodeResult> res) {
    if (static_cast<size_t>(res.size()) == results_.size()) {
      throw std::logic_error("The number of the results doesn't match one specified at "
                             "promise's construction.");
    }

    std::lock_guard lg(mtx_);
    for (int i = 0, n = res.size(); i < n; i++) {
      if (ready_mask_[i])
        throw std::logic_error("Entry already set.");
      results_[i] = std::move(res[i]);
    }
    ready_indices_.resize(res.size());
    std::iota(ready_indices_.begin(), ready_indices_.end(), 0);

    cv_any_.notify_all();
  }

  std::mutex mtx_;
  std::condition_variable cv_any_;

  std::atomic_flag has_future_ = ATOMIC_FLAG_INIT;
  std::vector<DecodeResult> results_;
  std::vector<int> ready_indices_;
  std::vector<uint8_t> ready_mask_;  // avoid vector<bool>
  size_t last_checked_ = 0;

  static thread_local std::deque<std::shared_ptr<DecodeResultsSharedState>> free_;
};

thread_local std::deque<std::shared_ptr<DecodeResultsSharedState>>
    DecodeResultsSharedState::free_;

FutureDecodeResults DecodeResultsPromise::get_future() const {
  std::atomic_flag flag;
  if (impl_->has_future_.test_and_set())
    throw std::logic_error("There's already a future associated with this promise.");
  return FutureDecodeResults(impl_);
}

FutureDecodeResults::FutureDecodeResults(std::shared_ptr<DecodeResultsSharedState> impl)
: impl_(std::move(impl)) {}


FutureDecodeResults::~FutureDecodeResults() {
  if (impl_) {
    #pragma GCC diagnostic push
  #ifdef __clang__
    #pragma GCC diagnostic ignored "-Wexceptions"
  #else
    #pragma GCC diagnostic ignored "-Wterminate"
  #endif
    if (impl_->ready_indices_.size() != impl_->results_.size())
      throw std::logic_error("Deferred results incomplete");
    #pragma GCC diagnostic pop
    impl_->reset();
    DecodeResultsSharedState::put(std::move(impl_));
  }
}

void FutureDecodeResults::wait_all() const {
  impl_->wait_all();
}

cspan<int> FutureDecodeResults::wait_new() const {
  return impl_->wait_new();
}

void FutureDecodeResults::wait_one(int index) const {
  return impl_->wait_one(index);
}


int FutureDecodeResults::num_samples() const {
  return impl_->results_.size();
}

cspan<DecodeResult> FutureDecodeResults::get_all_ref() const {
  wait_all();
  return make_span(impl_->results_);
}

std::vector<DecodeResult> FutureDecodeResults::get_all_copy() const {
  wait_all();
  return impl_->results_;
}

DecodeResult FutureDecodeResults::get_one(int index) const {
  wait_one(index);
  return impl_->results_[index];
}

void DecodeResultsPromise::set(int index, DecodeResult res) {
  impl_->set(index, std::move(res));
}

void DecodeResultsPromise::set_all(span<DecodeResult> res) {
  impl_->set_all(res);
}

int DecodeResultsPromise::num_samples() const {
  return impl_->results_.size();
}

DecodeResultsPromise::DecodeResultsPromise(int num_samples) {
  impl_ = DecodeResultsSharedState::get();
  impl_->init(num_samples);
}

}  // namespace imgcodec
}  // namespace dali
