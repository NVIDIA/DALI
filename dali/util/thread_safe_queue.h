// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_THREAD_SAFE_QUEUE_H_
#define DALI_UTIL_THREAD_SAFE_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace dali {

template<typename T>
class ThreadSafeQueue {
 public:
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(lock_);
      queue_.push(std::move(item));
    }
    cond_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock{lock_};
    cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
    if (interrupt_) {
      return {};
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  const T& peek() {
    static const auto int_return = T{};
    std::unique_lock<std::mutex> lock{lock_};
    cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
    if (interrupt_) {
      return int_return;
    }
    return queue_.front();
  }

  bool empty() const {
    return queue_.empty();
  }

  typename std::queue<T>::size_type size() const {
    return queue_.size();
  }

  void shutdown() {
    interrupt_ = true;
    cond_.notify_all();
  }

 private:
  std::queue<T> queue_;
  std::mutex lock_;
  std::condition_variable cond_;
  bool interrupt_ = false;
};

}  // namespace dali

#endif  // DALI_UTIL_THREAD_SAFE_QUEUE_H_
