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

#ifndef DALI_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
#define DALI_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

#define SEQUENCEWRAPPER_SUPPORTED_TYPES (float, uint8_t)

// Struct that Loader::ReadOne will read
struct SequenceWrapper {
 public:
  SequenceWrapper()
  : started_(false) {}

  void initialize(int count, int height, int width, int channels, DALIDataType dtype) {
    this->count = count;
    this->height = height;
    this->width = width;
    this->channels = channels;
    this->dtype = dtype;

    TYPE_SWITCH(dtype, type2id, OutputType, SEQUENCEWRAPPER_SUPPORTED_TYPES, (
        sequence.set_type(TypeInfo::Create<OutputType>());
      ), DALI_FAIL(make_string("Not supported output type:", dtype));); // NOLINT

    sequence.Resize({count, height, width, channels});
    timestamps.clear();
    timestamps.reserve(count);

    int dev;
    CUDA_CALL(cudaGetDevice(&dev));
    std::unique_lock<std::mutex> lock{started_lock_};
    if (started_) {
      CUDA_CALL(cudaEventDestroy(event_));
    }
    CUDA_CALL(cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming));
    started_ = false;
  }

  ~SequenceWrapper() {
    std::unique_lock<std::mutex> lock{started_lock_};
    if (started_) {
      try {
        CUDA_CALL(cudaEventDestroy(event_));
      } catch (const std::exception &) {
        // Something went wrong with releasing resources. We'd better die now.
        std::terminate();
      }
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
    LOG_LINE << event_ << " waiting for sequence event" << std::endl;
    CUDA_CALL(cudaEventSynchronize(event_));
    LOG_LINE << event_ << " synchronized!" << std::endl;
  }

  Tensor<GPUBackend> sequence;
  int count;
  int height;
  int width;
  int channels;
  int label;
  vector<double> timestamps;
  int first_frame_idx;
  DALIDataType dtype;

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

#endif  // DALI_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
