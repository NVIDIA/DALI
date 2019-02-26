// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_
#define DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_

#include <condition_variable>
#include <mutex>
#include <queue>

#include "dali/pipeline/executor/queue_metadata.h"

namespace dali {

// // Policy that passes Queueing/Buffering indexes between stages, handling required synchronization
// struct QueuePolicy {
//   // Initialize the policy during Executor::Build();
//   void InitializeQueues(
//       const std::array<int, static_cast<int>(DALIOpType::COUNT)> &stage_queue_depths);
//   // Acquire Queue indexes for given stage
//   QueueIdxs AcquireIdxs(DALIOpType stage);
//   // Finish stage and release the indexes. Not called by the last stage, as it "returns" outputs
//   void ReleaseIdxs(DALIOpType stage, QueueIdxs idxs);
//   // Called by the last stage - mark the Queue idxs as ready to be used as output
//   void QueueOutputIdxs(QueueIdxs idxs);
//   // Get the indexes of ready outputs and mark them as in_use by the user
//   OutputIdxs UseOutputIdxs();
//   // Release currently used output
//   void ReleaseOutputIdxs();
//   // Wake all waiting threads and skip further execution due to error
//   void SignalError();
//   // Returns true if we signaled an error previously
//   bool IsErrorSignaled();
// };


// Each stage requires ready buffers from previous stage and free buffers from current stage
struct UniformQueuePolicy {
  void InitializeQueues(
      const std::array<int, static_cast<int>(DALIOpType::COUNT)> &stage_queue_depths) {
    // TODO(klecki): enforce uniform here:
    DALI_ENFORCE(stage_queue_depths[(int)DALIOpType::CPU] == stage_queue_depths[(int)DALIOpType::MIXED] &&
                     stage_queue_depths[(int)DALIOpType::MIXED] == stage_queue_depths[(int)DALIOpType::GPU],
                 "This policy does not support splited queues");

    // All buffers start off as free
    for (int i = 0; i < stage_queue_depths[(int)DALIOpType::CPU]; ++i) {
      free_queue_.push(i);
    }
  }

  QueueIdxs AcquireIdxs(DALIOpType stage) {
    if (exec_error_) {
      return QueueIdxs{-1};
    }
    if (stage == DALIOpType::SUPPORT) {
      // Block until there is a free buffer to use
      std::unique_lock<std::mutex> lock(free_mutex_);
      while (free_queue_.empty() && !exec_error_) {
        free_cond_.wait(lock);
      }
      if (exec_error_) {
        return QueueIdxs{-1}; // We return antyhing due to exec error
      }
      int queue_idx = free_queue_.front();
      free_queue_.pop();
      return QueueIdxs{queue_idx};
      // lock.unlock();
    }

    std::unique_lock<std::mutex> lock(stage_work_mutex_[static_cast<int>(stage)]);
    auto queue_idx = stage_work_queue_[static_cast<int>(stage)].front();
    stage_work_queue_[static_cast<int>(stage)].pop();
    return QueueIdxs{queue_idx};
  }

  void ReleaseIdxs(DALIOpType stage, QueueIdxs idxs) {
    if (HasNextStage(stage)) {
      auto next_stage = NextStage(stage);
      std::unique_lock<std::mutex> lock(stage_work_mutex_[static_cast<int>(next_stage)]);
      stage_work_queue_[static_cast<int>(next_stage)].push(idxs[stage]);
    }
  }

  void QueueOutputIdxs(QueueIdxs idxs) {
    // We have to give up the elements to be occupied
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_queue_.push(idxs[DALIOpType::GPU]);
    lock.unlock();
    ready_cond_.notify_all();
  }

  OutputIdxs UseOutputIdxs() {
    if (exec_error_) {
      // std::unique_lock<std::mutex> errors_lock(errors_mutex_);
      // std::string error = errors_.empty() ? "Unknown error" : errors_.front();
      // throw std::runtime_error(error);
    }

    // Block until the work for a batch has been issued.
    // Move the queue id from ready to in_use
    std::unique_lock<std::mutex> lock(ready_mutex_);
    while (ready_queue_.empty() && !exec_error_) {
      ready_cond_.wait(lock);
    }
    if (exec_error_) {
      return OutputIdxs{-1};
    }
    int output_idx = ready_queue_.front();
    ready_queue_.pop();
    in_use_queue_.push(output_idx);
    lock.unlock();
    return OutputIdxs{output_idx};
  }

  void ReleaseOutputIdxs() {
    // Mark the last in-use buffer as free and signal
    // to waiting threads
    if (!in_use_queue_.empty()) { // TODO(klecki): in_use_queue should be guarded
      std::unique_lock<std::mutex> lock(free_mutex_);
      free_queue_.push(in_use_queue_.front());
      in_use_queue_.pop();
      lock.unlock();
      free_cond_.notify_one();
    }
  }

  void SignalError() {
    exec_error_ = true;
    ready_cond_.notify_all();
    free_cond_.notify_all();
  }

  bool IsErrorSignaled() {
    return exec_error_;
  }
 protected:
  bool exec_error_ = false;

 private:
  std::queue<int> ready_queue_, free_queue_, in_use_queue_;
  std::mutex ready_mutex_, free_mutex_;
  std::condition_variable ready_cond_, free_cond_;

  std::array<std::queue<int>, static_cast<int>(DALIOpType::COUNT)> stage_work_queue_;
  std::array<std::mutex, static_cast<int>(DALIOpType::COUNT)> stage_work_mutex_;

};


struct AsyncUniformQueuePolicy : public UniformQueuePolicy {
  // Acquire Queue indexes for given stage
  QueueIdxs AcquireIdxs(DALIOpType stage) {
    switch(stage) {
      // This is the start of RunCPU()
      case DALIOpType::SUPPORT: {
        std::unique_lock<std::mutex> lock(cpu_mutex_);
        DALI_ENFORCE(cpu_work_counter_ > 0, "Internal error, thread has no cpu work.");
        --cpu_work_counter_;
        lock.unlock();
        break;
      }
      case DALIOpType::MIXED: {
        // Block until there is mixed work to do
        std::unique_lock<std::mutex> lock(mixed_mutex_);
        while (mixed_work_counter_ == 0 && !exec_error_) {
          mixed_work_cv_.wait(lock);
        }
        --mixed_work_counter_;
        lock.unlock();
        if (exec_error_) {
          gpu_work_cv_.notify_all();
          // return;
        }
        break;
      }
      case DALIOpType::GPU: {
        // Block until there is gpu work to do
        std::unique_lock<std::mutex> lock(gpu_mutex_);
        while (gpu_work_counter_ == 0 && !exec_error_) {
          gpu_work_cv_.wait(lock);
        }
        --gpu_work_counter_;
        lock.unlock();
        // if (exec_error_)
        //   return;
        break;
      }
      default:
      // Other cases handled by base class
      break;
    }
    return UniformQueuePolicy::AcquireIdxs(stage);
  }

  // Finish stage and release the indexes. Not called by the last stage, as it "returns" outputs
  void ReleaseIdxs(DALIOpType stage, QueueIdxs idxs) {
    switch(stage) {
      // This is the end of RunCPU()
      case DALIOpType::CPU: {
        // Mark that there is now mixed work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> mixed_lock(mixed_mutex_);
        ++mixed_work_counter_;
        mixed_work_cv_.notify_one();
        break;
      }
      case DALIOpType::MIXED: {
        // Mark that there is now gpu work to do
        // and signal to any threads that are waiting
        std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
        ++gpu_work_counter_;
        gpu_work_cv_.notify_one();
        gpu_lock.unlock();
        break;
      }
      default:
      // Other cases handled by base class
      break;
    }
    UniformQueuePolicy::ReleaseIdxs(stage, idxs);
  }

 protected:
  // TODO(klecki): hack for old async pipeline
  int cpu_work_counter_ = 0;
  std::mutex cpu_mutex_;
 private:
  int mixed_work_counter_ = 0, gpu_work_counter_ = 0;
  std::mutex mixed_mutex_, gpu_mutex_;
  std::condition_variable mixed_work_cv_, gpu_work_cv_;
};

// Ready buffers from previous stage imply that we can process corresponding buffers from current
// stage
struct SeparateQueuePolicy {
  void InitializeQueues(
      const std::array<int, static_cast<int>(DALIOpType::COUNT)> &stage_queue_depths) {
    for (int stage = 0; stage < static_cast<int>(DALIOpType::COUNT); stage++) {
      for (int i = 0; i < stage_queue_depths[stage]; i++) {
        stage_free_[stage].push(i);
      }
    }
  }

  QueueIdxs AcquireIdxs(DALIOpType stage) {
    std::cout << "Try Acquire for " << to_string(stage) << std::endl;
    if (exec_error_) {
      return QueueIdxs{-1};
    }
    QueueIdxs result(0);
    // We dine with the philosophers
    std::cout << "Acquire for " << to_string(stage) << std::endl;

    int current_stage = static_cast<int>(stage);
    // We actually have a previous stage
    if (HasPreviousStage(stage)) {
      int previous_stage = static_cast<int>(PreviousStage(stage));
      std::unique_lock<std::mutex> ready_previous_lock(stage_ready_mutex_[previous_stage]);
      stage_ready_cv_[previous_stage].wait(ready_previous_lock, [previous_stage, this]() {
        return !stage_ready_[previous_stage].empty() || exec_error_;
      });
      if (exec_error_) {
        std::cout << "WUT" << std::endl;
        return QueueIdxs{-1};
      }
      result[static_cast<DALIOpType>(previous_stage)] = stage_ready_[previous_stage].front();
      stage_ready_[previous_stage].pop();
      // We are the only ones waiting for the lock, so we do not try to wake anyone
    }
    // There always is a current stage
    {
      std::unique_lock<std::mutex> free_current_lock(stage_free_mutex_[current_stage]);
      stage_free_cv_[current_stage].wait(free_current_lock, [current_stage, this]() {
        return !stage_free_[current_stage].empty() || exec_error_;
      });
      if (exec_error_) {
        return QueueIdxs{-1};
      }
      result[stage] = stage_free_[current_stage].front();
      stage_free_[current_stage].pop();
      // As above? TODO(klecki): Where do we wake anyone
    }
    std::cout << "Acquired for " << to_string(stage) << " " << result << std::endl;
    return result;
  }

  void ReleaseIdxs(DALIOpType stage, QueueIdxs idxs) {
    int current_stage = static_cast<int>(stage);
    std::cout << "Releasing for " << to_string(stage) << " " << idxs << std::endl;
    if (HasPreviousStage(stage)) {
      int previous_stage = static_cast<int>(PreviousStage(stage));
      // We always can just release the consumed buffer
      std::unique_lock<std::mutex> free_previous_lock(stage_free_mutex_[previous_stage]);
      stage_free_[previous_stage].push(idxs[static_cast<DALIOpType>(previous_stage)]);
    }
    // We freed buffer, so we notfiy the previous stage it can continue it's work
    if (HasPreviousStage(stage)) {
      int previous_stage = static_cast<int>(PreviousStage(stage));
      stage_free_cv_[previous_stage].notify_one();
    }
    {
      std::unique_lock<std::mutex> ready_current_lock(stage_ready_mutex_[current_stage]);
      stage_ready_[current_stage].push(idxs[stage]);
    }
    stage_ready_cv_[current_stage].notify_one();
  }

  void QueueOutputIdxs(QueueIdxs idxs) {
    std::cout << "Queueing outputs " << idxs << std::endl;
    std::unique_lock<std::mutex> ready_output_lock(ready_output_mutex_);
    ready_output_queue_.push({idxs[DALIOpType::MIXED], idxs[DALIOpType::GPU]});
    ready_output_lock.unlock();
    ready_output_cv_.notify_all();
  }

  // TODO(klecki): Error handling
  OutputIdxs UseOutputIdxs() {
    // Block until the work for a batch has been issued.
    // Move the queue id from ready to in_use
    std::unique_lock<std::mutex> ready_lock(ready_output_mutex_);
    // while (ready_output_queue_.empty() && !exec_error_) {
    //   ready_output_cv_.wait(ready_lock);
    //   if (exec_error_) {
    //     break;
    //   }
    // }
    ready_output_cv_.wait(ready_lock,
                          [this]() { return !ready_output_queue_.empty() || exec_error_; });
    if (exec_error_) {
      return OutputIdxs{-1, -1};
    }
    // if (exec_error_) {
    //   std::unique_lock<std::mutex> errors_lock(errors_mutex_);
    //   std::string error = errors_.empty() ? "Unknown error" : errors_.front();
    //   throw std::runtime_error(error);
    // }
    auto output_idx = ready_output_queue_.front();
    ready_output_queue_.pop();
    std::cout << "Marking as in use: " << output_idx.mixed << ", " << output_idx.gpu << std::endl;
    in_use_queue_.push(output_idx);  // TODO(klecki) -this may cause some problems!!!
    ready_lock.unlock();
    return output_idx;
  }

  void ReleaseOutputIdxs() {
    // Mark the last in-use buffer as free and signal
    // to waiting threads
    if (!in_use_queue_.empty()) {
      auto mixed_idx = static_cast<int>(DALIOpType::MIXED);
      auto gpu_idx = static_cast<int>(DALIOpType::GPU);
      auto processed = in_use_queue_.front();  // TODO(klecki): this should be guarded as well
      in_use_queue_.pop();
      std::cout << "Releasing outputs: " << processed.mixed << ", " << processed.gpu << std::endl;
      {
        std::unique_lock<std::mutex> lock(stage_free_mutex_[mixed_idx]);
        stage_free_[mixed_idx].push(processed.mixed);
      }
      stage_free_cv_[mixed_idx].notify_one();
      {
        std::unique_lock<std::mutex> lock(stage_free_mutex_[gpu_idx]);
        stage_free_[gpu_idx].push(processed.gpu);
      }
      stage_free_cv_[gpu_idx].notify_one();
    }
  }

  void SignalError() {
    std::cout << "Signaling error" << std::endl;
    exec_error_ = true;
    ready_output_cv_.notify_all();
    free_cond_.notify_all();
  }

  bool IsErrorSignaled() {
    return exec_error_;
  }

 private:
  // For syncing free and ready buffers between stages
  std::array<std::mutex, static_cast<int>(DALIOpType::COUNT)> stage_free_mutex_;
  std::array<std::mutex, static_cast<int>(DALIOpType::COUNT)> stage_ready_mutex_;
  std::array<std::condition_variable, static_cast<int>(DALIOpType::COUNT)> stage_free_cv_;
  std::array<std::condition_variable, static_cast<int>(DALIOpType::COUNT)> stage_ready_cv_;

  // Buffers are rotated between being 'free', where the
  // pipeline is ok to fill them with data, 'ready', where
  // they are already full of prepared data, and 'in-use',
  // where the user currently owns that buffer. A buffer
  // is marked as in-use when it is returned as and output.
  // The buffer is then returned the the ready queue the
  // next time Ouputs() is called.
  std::array<std::queue<int>, static_cast<int>(DALIOpType::COUNT)> stage_free_;
  std::array<std::queue<int>, static_cast<int>(DALIOpType::COUNT)> stage_ready_;

  std::condition_variable ready_output_cv_, free_cond_;
  // Output ready and in_use mutexes and queues
  std::mutex ready_output_mutex_, in_use_mutex_;

  std::queue<OutputIdxs> ready_output_queue_;
  std::queue<OutputIdxs> in_use_queue_;

  bool exec_error_ = false;
};

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_QUEUE_POLICY_H_