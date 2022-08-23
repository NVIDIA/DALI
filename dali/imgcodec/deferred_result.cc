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
#include <set>
#include <condition_variable>
#include "dali/imgcodec/image_decoder_interfaces.h"

namespace dali {
namespace imgcodec {

class DeferredDecodeResults::Impl {
 public:
  static Impl *get() {
    if (free_.empty())
      return new Impl();

    Impl *ret = free_.back().release();
    free_.pop_back();
    return ret;
  }

  static void put(Impl *impl) {
    free_.emplace_back(impl);
  }

  void clear() {
    ready_.clear();
    results_.clear();
  }

  void wait(int index) {
    if (ready_.find(index) != ready_.end())
      return;
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [&]() {
      return ready_.find(index) != ready_.end();
    });
  }

  void wait_all() {
    if (ready_.size() == results_.size())
      return;
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [&]() {
      return ready_.size() == results_.size();
    });
  }

  void set(int index, DecodeResult res) {
    std::lock_guard lg(mtx_);
    results_[index] = std::move(res);
    ready_.insert(index);
    cv_.notify_all();
  }

  void set_all(span<const DecodeResult> results) {
    std::lock_guard lg(mtx_);
    for (int index = 0; index < results.size(); index++) {
      const auto &res = results[index];
      results_[index] = res;
      ready_.insert(index);
    }
    cv_.notify_all();
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  std::set<int> ready_;
  std::vector<DecodeResult> results_;

  static thread_local std::deque<std::unique_ptr<Impl>> free_;
};

thread_local std::deque<std::unique_ptr<DeferredDecodeResults::Impl>>
    DeferredDecodeResults::Impl::free_;


DeferredDecodeResults::DeferredDecodeResults(int num_samples) {
  impl_ = Impl::get();
  impl_->results_.resize(num_samples);
}

DeferredDecodeResults::~DeferredDecodeResults() {
  if (impl_) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wexceptions"  // we want it to terminate _with a message_
    if (impl_->ready_.size() != impl_->results_.size())
      throw std::logic_error("Deferred results incomplete");
    #pragma GCC diagnostic pop
    Impl::put(impl_);
    impl_ = nullptr;
  }
}

void DeferredDecodeResults::wait(int index) const {
  impl_->wait(index);
}

void DeferredDecodeResults::wait_all() const {
  impl_->wait_all();
}

DecodeResult DeferredDecodeResults::get(int index) const {
  wait(index);
  return impl_->results_[index];
}

span<DecodeResult> DeferredDecodeResults::get_all() const {
  wait_all();
  return make_span(impl_->results_);
}

void DeferredDecodeResults::set(int index, DecodeResult res) {
  impl_->set(index, std::move(res));
}

void DeferredDecodeResults::set_all(span<const DecodeResult> res) {
  impl_->set_all(res);
}

int DeferredDecodeResults::num_samples() const {
  return impl_->results_.size();
}

}  // namespace imgcodec
}  // namespace dali
