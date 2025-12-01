// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_event.h"
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
  SequenceWrapper() = default;

  void initialize(int count, int max_count, int height, int width, int channels,
                  DALIDataType dtype) {
    this->count = count;
    this->max_count = max_count;
    this->height = height;
    this->width = width;
    this->channels = channels;
    this->dtype = dtype;

    timestamps.clear();
    timestamps.reserve(max_count);

    if (!event_) {
      event_ = CUDAEvent::CreateWithFlags(cudaEventBlockingSync | cudaEventDisableTiming);
    }
  }

  void set_started(cudaStream_t stream) {
    CUDA_CALL(cudaEventRecord(event_, stream));
  }

  void wait() const {
    if (event_) {
      LOG_LINE << event_ << " waiting for sequence event" << std::endl;
      CUDA_CALL(cudaEventSynchronize(event_));
      LOG_LINE << event_ << " synchronized!" << std::endl;
    }
  }

  TensorShape<3> frame_shape() const {
    return TensorShape<3>{height, width, channels};
  }

  TensorShape<4> shape() const {
    return TensorShape<4>{max_count, height, width, channels};
  }

  Tensor<GPUBackend> sequence;

  int count = -1;
  int max_count = -1;
  int height = -1;
  int width = -1;
  int channels = -1;
  int label = -1;
  vector<double> timestamps;
  int first_frame_idx = -1;
  DALIDataType dtype = DALI_NO_TYPE;
  std::function<void(void)> read_sample_f;

 private:
  CUDAEvent event_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NVDECODER_SEQUENCEWRAPPER_H_
