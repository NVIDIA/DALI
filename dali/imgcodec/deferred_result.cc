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

  void wait() {
    if (results_ready_ == results_.size())
      return;
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [&]() {
      return results_ready_ == results_.size();
    });
  }

  void set(int index, DecodeResult res) {
    {
      std::lock_guard lg(mtx_);
      results_[index] = std::move(res);
      if (++results_ready_ == results_.size())
        cv_.notify_all();
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  size_t results_ready_ = 0;
  std::vector<DecodeResult> results_;

  static thread_local std::deque<std::unique_ptr<Impl>> free_;
};

thread_local std::deque<std::unique_ptr<DeferredDecodeResults::Impl>>
    DeferredDecodeResults::Impl::free_;


DeferredDecodeResults::DeferredDecodeResults(int num_samples) {
  impl_ = Impl::get();
  impl_->results_.resize(num_samples);
  impl_->results_ready_ = 0;
}

DeferredDecodeResults::~DeferredDecodeResults() {
  if (impl_) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wexceptions"  // we want it to terminate _with a message_
    if (impl_->results_ready_ != impl_->results_.size())
      throw std::logic_error("Deferred results incomplete");
    #pragma GCC diagnostic pop
    Impl::put(impl_);
    impl_ = nullptr;
  }
}

void DeferredDecodeResults::wait() const {
  impl_->wait();
}

int DeferredDecodeResults::num_samples() const {
  return impl_->results_.size();
}

span<DecodeResult> DeferredDecodeResults::get_all() const {
  wait();
  return make_span(impl_->results_);
}

DecodeResult DeferredDecodeResults::get_one(int index) const {
  wait();
  return impl_->results_[index];
}

void DeferredDecodeResults::set(int index, DecodeResult res) {
  impl_->set(index, std::move(res));
}


}  // namespace imgcodec
}  // namespace dali
