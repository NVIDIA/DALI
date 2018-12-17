// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
#define DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_

#include <condition_variable>
#include <mutex>
#include <thread>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

// Struct that Loader::ReadOne will read
struct SequenceWrapper {
 public:
  SequenceWrapper()
  : started_(false) {}

  void initialize(int count, int height, int width, int channels) {
    this->count = count;
    this->height = height;
    this->width = width;
    this->channels = channels;
    // TODO(spanev) Handle other types
    sequence.set_type(TypeInfo::Create<float>());
    sequence.Resize({count, height, width, channels});

    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    if (started_) {
      CUDA_CALL(cudaEventDestroy(event_));
    }
    CUDA_CALL(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    started_ = false;
  }

  ~SequenceWrapper() {
    if (started_) {
      CUDA_CALL(cudaEventDestroy(event_));
    }
  }

  void set_started(cudaStream_t stream) {
    CUDA_CALL(cudaEventRecord(event_, stream));
    LOG_LINE << event_ << " recorded with stream " << stream << std::endl;
    {
      std::unique_lock<std::mutex> lock{started_lock_};
      started_ = true;
    }
    started_cv_.notify_one();
  }

  void wait() const {
    LOG_LINE << event_ << " wait to start" << std::endl;
    wait_until_started_();
    CUDA_CALL(cudaEventSynchronize(event_));
    LOG_LINE << event_ << " synchronized!" << std::endl;
  }

  Tensor<GPUBackend> sequence;
  int count;
  int height;
  int width;
  int channels;

 private:
  void wait_until_started_() const {
      std::unique_lock<std::mutex> lock{started_lock_};
      started_cv_.wait(lock, [&](){return started_;});
  }

  mutable std::mutex started_lock_;
  mutable std::condition_variable started_cv_;
  cudaEvent_t event_;
  bool started_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
